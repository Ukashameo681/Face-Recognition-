import cv2
import mediapipe as mp


class FaceDetector:
    def __init__(self, max_num_faces=1, min_detection_confidence=0.5):
        # Initialize MediaPipe FaceMesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            min_detection_confidence=min_detection_confidence
        )

        # Initialize drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    def detect_faces(self, frame):
        """
        Detect faces in the given frame.

        Args:
            frame (numpy.ndarray): The input frame from a video or image.

        Returns:
            rgb_frame (numpy.ndarray): The RGB converted frame.
            result (FaceMesh result): Face detection and landmark results.
        """
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect face and landmarks
        result = self.face_mesh.process(rgb_frame)

        return rgb_frame, result

    def draw_landmarks(self, frame, result):
        """
        Draw the facial landmarks on the detected face.

        Args:
            frame (numpy.ndarray): The original frame from the webcam.
            result (FaceMesh result): The result from the face detection.

        Returns:
            frame (numpy.ndarray): The frame with landmarks drawn.
        """
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                # Draw landmarks on the frame
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.drawing_spec,
                    connection_drawing_spec=self.drawing_spec
                )
        return frame


# Main code to use the FaceDetector class
if __name__ == "__main__":
    # Initialize video capture from the webcam
    cap = cv2.VideoCapture(0)

    # Create an instance of the FaceDetector class
    face_detector = FaceDetector(max_num_faces=1, min_detection_confidence=0.5)

    while True:
        ret, frame = cap.read()

        # Detect faces and get results
        rgb_frame, result = face_detector.detect_faces(frame)

        # Draw landmarks on the detected face
        frame_with_landmarks = face_detector.draw_landmarks(frame, result)

        # Display the frame with landmarks
        cv2.imshow('Facial Landmarks Detection', frame_with_landmarks)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()
