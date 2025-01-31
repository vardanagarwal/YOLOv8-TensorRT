"""
    This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
"""

from __future__ import print_function

import numpy as np
from .association import *
import sys


def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h  # scale is just area
    r = w / float(h + 1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score == None:
        return np.array(
            [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]
        ).reshape((1, 4))
    else:
        return np.array(
            [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]
        ).reshape((1, 5))


def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """

    count = 0

    def __init__(self, bbox, delta_t=3, orig=False):
        """
        Initialises a tracker using initial bounding box.
        bbox format: [x1, y1, x2, y2, score, class_id, classify]
        """
        # define constant velocity model
        if not orig:
            from .kalmanfilter import KalmanFilterNew as KalmanFilter

            self.kf = KalmanFilter(dim_x=7, dim_z=4)
        else:
            from filterpy.kalman import KalmanFilter

            self.kf = KalmanFilter(dim_x=7, dim_z=4)

        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        )

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[
            4:, 4:
        ] *= 1000.0  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        """
        NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of 
        function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a 
        fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]), let's bear it for now.
        """
        self.class_id = bbox[5] if len(bbox) > 5 else None
        self.classify = bbox[6] if len(bbox) > 6 else None
        self.score = bbox[4] if len(bbox) > 4 else None
        if isinstance(bbox, (list, tuple)):
            bbox = bbox[:4]
        else:
            bbox = np.array([-1, -1, -1, -1, -1])
        self.last_observation = np.array(bbox)
        # self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
        self.observations = dict()
        self.history_observations = []
        self.velocity = None
        self.delta_t = delta_t

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        if bbox is not None:
            # Extract just the numeric parts (bbox, score, class)
            if isinstance(bbox, (list, np.ndarray)):
                # Find the first non-numeric element
                numeric_part = []
                for x in bbox:
                    if isinstance(x, (int, float, np.number)):
                        numeric_part.append(x)
                    else:
                        break
                self.last_observation = np.array(numeric_part)
            else:
                self.last_observation = np.array(bbox)

            if self.last_observation.sum() >= 0:  # no previous observation
                previous_box = None
                for i in range(self.delta_t):
                    dt = self.delta_t - i
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age - dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                """
                    Estimate the track speed direction with observations \Delta t steps away
                """
                self.velocity = speed_direction(previous_box, bbox)

            if len(bbox) > 4:
                self.score = bbox[4]
            if len(bbox) > 5:
                self.class_id = bbox[5]
            if len(bbox) > 6:
                self.classify = bbox[6]

            """
                Insert new observations. This is a ugly way to maintain both self.observations
                and self.history_observations. Bear it for the moment.
            """
            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            self.kf.update(convert_bbox_to_z(bbox))
        else:
            self.kf.update(bbox)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate as [x1,y1,x2,y2]
        """
        if self.last_observation is not None and isinstance(
            self.last_observation, (list, tuple)
        ):
            # Extract just the bbox coordinates if last_observation contains additional info
            return np.array([self.last_observation[:4]])
        return self.last_observation.reshape((1, -1))


"""
    We support multiple ways for association cost calculation, by default
    we use IoU. GIoU may have better performance in some situations. We note 
    that we hardly normalize the cost by all methods to (0,1) which may not be 
    the best practice.
"""
ASSO_FUNCS = {
    "iou": iou_batch,
    "giou": giou_batch,
    "ciou": ciou_batch,
    "diou": diou_batch,
    "ct_dist": ct_dist,
}


class OCSort(object):
    def __init__(
        self,
        det_thresh,  # Minimum confidence threshold for detections to be considered
        max_age=30,  # Maximum frames an unmatched track can survive before being removed
        min_hits=3,  # Minimum detection hits needed to establish a track
        iou_threshold=0.3,  # IoU threshold for matching detections to existing tracks
        delta_t=3,  # Time step between frames for motion prediction
        asso_func="iou",  # Association function type (IoU based matching)
        inertia=0.2,  # Weight of motion prediction in tracking (higher = more motion influence)
        use_byte=False,  # Whether to use ByteTrack-style association
    ):
        """
        OCSort - Multi-Object Tracking Algorithm

        Parameters:
        -----------
        det_thresh : float, default=0.25
            Detection confidence threshold. Only detections with confidence above this
            value will be considered for tracking. Range: [0.0, 1.0]
            Lower values increase sensitivity but may include more false positives.

        max_age : int, default=6
            Maximum number of frames a track can remain unmatched before being deleted.
            Higher values help maintain tracks through occlusions but may retain false tracks longer.

        min_hits : int, default=2
            Minimum number of detection matches required to initialize a new track.
            Higher values create more stable tracks but may miss brief appearances.

        iou_threshold : float, default=0.3
            Minimum Intersection over Union (IoU) required to match detection to track.
            Range: [0.0, 1.0]. Higher values enforce stricter matching criteria.

        delta_t : int, default=1
            Time step between frames for motion prediction calculations.
            Typically kept at 1 for consecutive frame processing.

        asso_func : str, default="iou"
            Association function type for matching detections to tracks.
            Currently supported: "iou" (Intersection over Union based matching)

        inertia : float, default=0.2
            Weight given to motion prediction in tracking process.
            Range: [0.0, 1.0]. Higher values create smoother tracks but less responsive to sudden changes.

        use_byte : bool, default=False
            Whether to use ByteTrack-style association strategy.
            When True, enables additional processing of low-confidence detections.

        Notes:
        ------
        Tuning Recommendations:
        - Fast moving objects: Increase inertia, decrease iou_threshold
        - Crowded scenes: Increase det_thresh and min_hits
        - Heavy occlusions: Increase max_age
        - Reduce false tracks: Increase min_hits and det_thresh

        Returns:
        --------
        tracker : OCSort
            Initialized OCSort tracking object
        """

        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = ASSO_FUNCS[asso_func]
        self.inertia = inertia
        self.use_byte = use_byte
        KalmanBoxTracker.count = 0

    def update(self, output_results, img_info, img_size):
        try:
            """
            Params:
            dets - a numpy array of detections in the format [[[x1,y1,x2,y2,score,class_id,classify],...],[[x1,y1,x2,y2,score,class_id,classify],...],...]
            Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
            Returns the a similar array, where the last column is the object ID.
            NOTE: The number of objects returned may differ from the number of detections provided.

            output_results - a numpy array of detections in the format
            [[x1,y1,x2,y2,score,class_id,classify],...]
            """
            if output_results is None:
                return np.empty((0, 7))

            self.frame_count += 1

            # Post-process detections
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]

            # scores = np.array([float(score_str.split(",")[0]) for score_str in scores.tolist()], dtype=object)
            img_h, img_w = img_info[0], img_info[1]
            scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
            bboxes /= scale

            # Include class_id and classify in dets
            dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)
            if output_results.shape[1] > 5:  # If class_id is provided
                dets = np.concatenate((dets, output_results[:, 5:]), axis=1)

            inds_low = scores > 0.1
            inds_high = scores < self.det_thresh
            inds_second = np.logical_and(
                inds_low, inds_high
            )  # self.det_thresh > score > 0.1, for second matching
            dets_second = dets[inds_second]  # detections for second matching
            remain_inds = scores > self.det_thresh
            dets = dets[remain_inds]
            # get predicted locations from existing trackers.

            trks = np.zeros((len(self.trackers), 5))
            to_del = []
            ret = []

            for t, trk in enumerate(trks):
                pos = self.trackers[t].predict()[0]
                trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
                if np.any(np.isnan(pos)):
                    to_del.append(t)

            trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
            for t in reversed(to_del):
                self.trackers.pop(t)

            velocities = np.array(
                [
                    trk.velocity if trk.velocity is not None else np.array((0, 0))
                    for trk in self.trackers
                ]
            )

            # last_boxes = np.array([trk.last_observation for trk in self.trackers])

            last_observation_array = []
            for trk in self.trackers:
                if isinstance(trk.last_observation, (list, np.ndarray)):
                    # Find the first non-numeric element
                    numeric_part = []
                    for x in trk.last_observation:
                        if isinstance(x, (int, float, np.number)):
                            numeric_part.append(x)
                        else:
                            break
                    last_observation_array.append(numeric_part)
                else:
                    last_observation_array.append(trk.last_observation)

            if last_observation_array:
                # Get the maximum length using len() instead of shape
                max_length = max(len(arr) for arr in last_observation_array)

                # Pad shorter arrays with -1
                _standardized = []
                for arr in last_observation_array:
                    if len(arr) < max_length:
                        padded = list(arr) + [-1] * (max_length - len(arr))
                        _standardized.append(padded)
                    else:
                        _standardized.append(arr)
                _standardized = np.array(_standardized)
                standardized = []
                max_dim = max(len(arr) for arr in _standardized)

                for arr in last_observation_array:
                    if len(arr) < max_dim:
                        # Create a padded array with -1 values
                        padded = np.full((len(arr), max_dim), -1, dtype=object)
                        # Copy original values
                        padded[:, : len(arr)] = arr
                        standardized.append(padded)
                    else:
                        standardized.append(arr)

            if last_observation_array:
                valid_row = next(
                    (row for row in last_observation_array if row[0] != -1),
                    last_observation_array[0],
                )
                target_length = len(valid_row)

                # Create array with appropriate dimensions
                last_boxes = np.full((len(last_observation_array), target_length), -1.0)

                # Fill in the data, padding shorter rows with -1
                for i, row in enumerate(last_observation_array):
                    if len(row) == target_length:
                        last_boxes[i] = row
                    else:
                        # Pad shorter rows with -1
                        last_boxes[i, : len(row)] = row
            else:
                last_boxes = np.array(last_observation_array)

            k_observations_list = []
            for trk in self.trackers:
                k_observations_list.append(
                    k_previous_obs(trk.observations, trk.age, self.delta_t)
                )

            if k_observations_list:
                max_len = max(
                    len(row) if isinstance(row, (list, np.ndarray)) else 5
                    for row in k_observations_list
                )

                numeric_length = 6
                # Create array with consistent shape
                k_observations = np.full((len(k_observations_list), max_len), -1.0)

                # Fill in the data
                for i, obs in enumerate(k_observations_list):
                    if isinstance(obs, (list, np.ndarray)) and obs[0] != -1:
                        # Only take the numeric values, excluding the dictionary
                        numeric_values = obs[:numeric_length]
                        k_observations[i, :numeric_length] = numeric_values
            else:
                k_observations = np.array(k_observations_list)

            """
                First round of association
            """
            matched, unmatched_dets, unmatched_trks = associate(
                dets, trks, self.iou_threshold, velocities, k_observations, self.inertia
            )
            for m in matched:
                self.trackers[m[1]].update(dets[m[0], :])

            """
                Second round of associaton by OCR
            """
            # BYTE association
            if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
                u_trks = trks[unmatched_trks]
                iou_left = self.asso_func(
                    dets_second, u_trks
                )  # iou between low score detections and unmatched tracks
                iou_left = np.array(iou_left)
                if iou_left.max() > self.iou_threshold:
                    """
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                    """
                    matched_indices = linear_assignment(-iou_left)
                    to_remove_trk_indices = []
                    for m in matched_indices:
                        det_ind, trk_ind = m[0], unmatched_trks[m[1]]
                        if iou_left[m[0], m[1]] < self.iou_threshold:
                            continue
                        self.trackers[trk_ind].update(dets_second[det_ind, :])
                        to_remove_trk_indices.append(trk_ind)
                    unmatched_trks = np.setdiff1d(
                        unmatched_trks, np.array(to_remove_trk_indices)
                    )

            if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
                left_dets = dets[unmatched_dets]
                left_trks = last_boxes[unmatched_trks]
                iou_left = self.asso_func(left_dets, left_trks)
                iou_left = np.array(iou_left)
                if iou_left.max() > self.iou_threshold:
                    """
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                    """
                    rematched_indices = linear_assignment(-iou_left)
                    to_remove_det_indices = []
                    to_remove_trk_indices = []
                    for m in rematched_indices:
                        det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                        if iou_left[m[0], m[1]] < self.iou_threshold:
                            continue
                        self.trackers[trk_ind].update(dets[det_ind, :])
                        to_remove_det_indices.append(det_ind)
                        to_remove_trk_indices.append(trk_ind)
                    unmatched_dets = np.setdiff1d(
                        unmatched_dets, np.array(to_remove_det_indices)
                    )
                    unmatched_trks = np.setdiff1d(
                        unmatched_trks, np.array(to_remove_trk_indices)
                    )

            for m in unmatched_trks:
                self.trackers[m].update(None)

            # create and initialise new trackers for unmatched detections
            for i in unmatched_dets:
                trk = KalmanBoxTracker(dets[i, :], delta_t=self.delta_t)
                self.trackers.append(trk)

            i = len(self.trackers)
            for trk in reversed(self.trackers):
                # Extract just the numeric parts (bbox, score, class)
                if isinstance(trk.last_observation, (list, np.ndarray)):
                    # Find the first non-numeric element
                    numeric_part = []
                    for x in trk.last_observation:
                        if isinstance(x, (int, float, np.number)):
                            numeric_part.append(x)
                        else:
                            break
                    trk.last_observation = np.array(numeric_part)
                else:
                    trk.last_observation = np.array(trk.last_observation)

                if trk.last_observation.sum() < 0:
                    d = trk.get_state()[0]
                else:
                    """
                    this is optional to use the recent observation or the kalman filter prediction,
                    we didn't notice significant difference here
                    """
                    d = trk.last_observation[:4]

                if (trk.time_since_update < 1) and (
                    trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
                ):
                    # +1 as MOT benchmark requires positive
                    ret_item = np.concatenate((d, [trk.id + 1]))
                    if hasattr(trk, "score") and trk.score is not None:
                        ret_item = np.concatenate((ret_item, [trk.score]))
                    if hasattr(trk, "class_id") and trk.class_id is not None:
                        ret_item = np.concatenate((ret_item, [trk.class_id]))
                    if hasattr(trk, "classify") and trk.classify is not None:
                        ret_item = np.concatenate((ret_item, [trk.classify]))

                    ret.append(ret_item.reshape(1, -1))
                i -= 1
                # remove dead tracklet
                if trk.time_since_update > self.max_age:
                    self.trackers.pop(i)

            if len(ret) > 0:
                standardized = []
                max_dim = max(arr.shape[1] for arr in ret)

                for arr in ret:
                    if arr.shape[1] < max_dim:
                        # Create a padded array with -1 values
                        padded = np.full((arr.shape[0], max_dim), -1, dtype=object)
                        # Copy original values
                        padded[:, : arr.shape[1]] = arr
                        standardized.append(padded)
                    else:
                        standardized.append(arr)

                # Now we can safely concatenate
                return np.concatenate(standardized)
            return []

        except Exception as e:
            print("WARNING Blank Detection: {}".format(e))
            return []

    def update_public(self, dets, cates, scores):
        self.frame_count += 1

        det_scores = np.ones((dets.shape[0], 1))
        dets = np.concatenate((dets, det_scores), axis=1)

        remain_inds = scores > self.det_thresh

        cates = cates[remain_inds]
        dets = dets[remain_inds]

        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            cat = self.trackers[t].cate
            trk[:] = [pos[0], pos[1], pos[2], pos[3], cat]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        velocities = np.array(
            [
                trk.velocity if trk.velocity is not None else np.array((0, 0))
                for trk in self.trackers
            ]
        )
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array(
            [
                k_previous_obs(trk.observations, trk.age, self.delta_t)
                for trk in self.trackers
            ]
        )

        matched, unmatched_dets, unmatched_trks = associate_kitti(
            dets,
            trks,
            cates,
            self.iou_threshold,
            velocities,
            k_observations,
            self.inertia,
        )

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            """
            The re-association stage by OCR.
            NOTE: at this stage, adding other strategy might be able to continue improve
            the performance, such as BYTE association by ByteTrack.
            """
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            left_dets_c = left_dets.copy()
            left_trks_c = left_trks.copy()

            iou_left = self.asso_func(left_dets_c, left_trks_c)
            iou_left = np.array(iou_left)
            det_cates_left = cates[unmatched_dets]
            trk_cates_left = trks[unmatched_trks][:, 4]
            num_dets = unmatched_dets.shape[0]
            num_trks = unmatched_trks.shape[0]
            cate_matrix = np.zeros((num_dets, num_trks))
            for i in range(num_dets):
                for j in range(num_trks):
                    if det_cates_left[i] != trk_cates_left[j]:
                        """
                        For some datasets, such as KITTI, there are different categories,
                        we have to avoid associate them together.
                        """
                        cate_matrix[i][j] = -1e6
            iou_left = iou_left + cate_matrix
            if iou_left.max() > self.iou_threshold - 0.1:
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold - 0.1:
                        continue
                    self.trackers[trk_ind].update(dets[det_ind, :])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(
                    unmatched_dets, np.array(to_remove_det_indices)
                )
                unmatched_trks = np.setdiff1d(
                    unmatched_trks, np.array(to_remove_trk_indices)
                )

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            trk.cate = cates[i]
            self.trackers.append(trk)
        i = len(self.trackers)

        for trk in reversed(self.trackers):
            if trk.last_observation.sum() > 0:
                d = trk.last_observation[:4]
            else:
                d = trk.get_state()[0]
            if trk.time_since_update < 1:
                if (self.frame_count <= self.min_hits) or (
                    trk.hit_streak >= self.min_hits
                ):
                    # id+1 as MOT benchmark requires positive
                    ret.append(
                        np.concatenate((d, [trk.id + 1], [trk.cate], [0])).reshape(
                            1, -1
                        )
                    )
                if trk.hit_streak == self.min_hits:
                    # Head Padding (HP): recover the lost steps during initializing the track
                    for prev_i in range(self.min_hits - 1):
                        prev_observation = trk.history_observations[-(prev_i + 2)]
                        ret.append(
                            (
                                np.concatenate(
                                    (
                                        prev_observation[:4],
                                        [trk.id + 1],
                                        [trk.cate],
                                        [-(prev_i + 1)],
                                    )
                                )
                            ).reshape(1, -1)
                        )
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 7))
