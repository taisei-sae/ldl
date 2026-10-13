import cv2
import torch
import numpy
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights
)

def main():
    cap = cv2.VideoCapture("data/20261004_131433_F_Gsn.AVI")
    if cap.isOpened():
        # Get the property of the video
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"FPS: {fps}")
        print(f"frames: {frame_count}")
        print(f"w: {w}")
        print(f"h: {h}")

        # Try getting the first frame
        is_successful, original_frame = cap.read()
        if is_successful:
            # prepare the model
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            model = fasterrcnn_resnet50_fpn_v2(weights=weights)
            model.eval()
            model.to("cuda")
            preprocess = weights.transforms()

            # prepare the frame image
            frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame)
            frame = frame.permute([2, 0, 1])
            # frame = preprocess(frame)
            frame = frame.to("cuda")

            # predict
            with torch.inference_mode():
                result = model([frame])

            # check the results
            boxes = result[0]["boxes"].cpu().numpy().astype(numpy.int32)
            labels = result[0]["labels"].cpu().numpy()
            scores = result[0]["scores"].cpu().numpy()

            res_frame = original_frame.copy()
            for label, score, box in zip(labels, scores, boxes):
                if score > 0.5:
                    cv2.rectangle(res_frame, [box[0], box[1]], [box[2], box[3]], (0, 255, 0))
                    cv2.putText(res_frame,
                                f"{weights.meta["categories"][label]}, {score:.3f}",
                                [box[0], box[1]],
                                cv2.FONT_HERSHEY_PLAIN,
                                1.0,
                                (0, 255, 0)
                                )

            cv2.imwrite("data/frame1.png", res_frame)
            
            cap.release()

    

if __name__ == "__main__":
    main()