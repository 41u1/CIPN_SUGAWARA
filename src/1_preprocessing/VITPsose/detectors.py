import torch
import numpy as np
import supervision as sv

from transformers import (
    AutoProcessor,
    RTDetrForObjectDetection,
    VitPoseForPoseEstimation,
)


class HumanDetector:
    def __init__(self, device="cpu"):
        self.device = device
        self.person_image_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
        self.model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365", device_map=self.device)
        self.results = None

    def process(self, image):
        inputs = self.person_image_processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        self.results = self.person_image_processor.post_process_object_detection(
            outputs, target_sizes=torch.tensor([(image.height, image.width)]), threshold=0.3
        )
    
    def get_person_boxes(self):
        result = self.results[0]
        # Human label refers 0 index in COCO dataset
        person_label_id = 0
        person_boxes_xyxy = result["boxes"][result["labels"] == person_label_id]
        person_boxes_xyxy = person_boxes_xyxy.cpu().numpy()

        return person_boxes_xyxy

    def convert_format(self, person_boxes_xyxy):
        # Convert boxes from VOC (x1, y1, x2, y2) to COCO (x1, y1, w, h) format
        person_boxes = person_boxes_xyxy.copy()
        person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
        person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]
        return person_boxes


class PoseDetector:
    def __init__(self, device="cpu"):
        self.device = device
        self.image_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
        self.model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple", device_map=self.device)
        self.pose_results = None
    
    def process(self, image, person_boxes):
        inputs = self.image_processor(image, boxes=[person_boxes], return_tensors="pt").to(self.device)
        # -----あまり関係ない-----
        # for vitpose-plus-base checkpoint we should additionaly provide dataset_index
        # to sepcify which MOE experts to use for inference
        if self.model.config.backbone_config.num_experts > 1:
            dataset_index = torch.tensor([0] * len(inputs["pixel_values"]))
            dataset_index = dataset_index.to(inputs["pixel_values"].device)
            inputs["dataset_index"] = dataset_index
        # -----------------------

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        self.pose_results = self.image_processor.post_process_pose_estimation(outputs, boxes=[person_boxes])

    def get_keypoints_array(self):
        image_pose_result = self.pose_results[0]
        person_pose = image_pose_result[0]
        key_points_list = [kp.cpu().numpy() for kp in person_pose["keypoints"]]
        return np.concatenate(key_points_list)
    
    def visualize_keypoints(self, image, person_boxes_xyxy):
        image_pose_result = self.pose_results[0]
        xy = [pose_result['keypoints'] for pose_result in image_pose_result]
        xy = torch.stack(xy).cpu().numpy()

        scores = [pose_result['scores'] for pose_result in image_pose_result]
        scores = torch.stack(scores).cpu().numpy()

        keypoints = sv.KeyPoints(xy=xy, confidence=scores)
        detections = sv.Detections(xyxy=person_boxes_xyxy)

        edge_annotator = sv.EdgeAnnotator(color=sv.Color.WHITE, thickness=3)
        vertex_annotator = sv.VertexAnnotator(color=sv.Color.BLUE, radius=4)
        bounding_box_annotator = sv.BoxAnnotator(
            color=sv.Color.GREEN, color_lookup=sv.ColorLookup.INDEX, thickness=2
        )

        annotated_frame = image.copy()

        # annotate boundg boxes
        annotated_frame = bounding_box_annotator.annotate(
            scene=image.copy(),
            detections=detections
        )

        # annotate edges and verticies
        annotated_frame = edge_annotator.annotate(
            scene=annotated_frame,
            key_points=keypoints
        )
        annotated_frame = vertex_annotator.annotate(
            scene=annotated_frame,
            key_points=keypoints
        )

        return annotated_frame