import cv2
import numpy as np

class WebcamFaultDetector:
    def __init__(self):
        """
        Initialize fault detector with interactive reference image selection
        """
        # Initialize webcam
        self.cap = cv2.VideoCapture(1)  # 0 for default webcam
        
        # Check if webcam opened successfully
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")
        
        # State flags
        self.reference_image = None
        self.reference_roi = None
        self.roi = None
        
        # Mouse interaction variables
        self.drawing = False
        self.ix, self.iy = -1, -1
        
        # Setup detection window
        cv2.namedWindow('Fault Detection', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Fault Detection', self.draw_rectangle)
    
    def draw_rectangle(self, event, x, y, flags, param):
        """
        Handle mouse events for drawing rectangles
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start drawing
            self.drawing = True
            self.ix, self.iy = x, y
        
        elif event == cv2.EVENT_MOUSEMOVE:
            # Create a copy of the frame to draw on
            if self.drawing:
                img_copy = self.current_frame.copy()
                cv2.rectangle(img_copy, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
                cv2.imshow('Fault Detection', img_copy)
        
        elif event == cv2.EVENT_LBUTTONUP:
            # Finish drawing
            self.drawing = False
            
            # Ensure valid rectangle
            x1, y1 = self.ix, self.iy
            x2, y2 = x, y
            
            # Normalize coordinates
            x_min = min(x1, x2)
            y_min = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            
            # Check if rectangle is large enough
            if w > 10 and h > 10:
                # If no reference image, set reference
                if self.reference_image is None:
                    self.reference_roi = (x_min, y_min, w, h)
                    self.reference_image = self.current_frame[y_min:y_min+h, x_min:x_min+w]
                # If reference is set but no ROI, set ROI
                elif self.roi is None:
                    self.roi = (x_min, y_min, w, h)
    
    def detect_faults(self, frame):
        """
        Detect faults by comparing ROI with reference image ROI.
        """
        if self.roi is None or self.reference_image is None:
            return {"fault_severity": "Not Ready", "difference_percentage": 0}
        
        # Extract current ROI
        x, y, w, h = self.roi
        current_roi = frame[y:y+h, x:x+w]
        
        # Resize current ROI to match reference ROI
        current_roi_resized = cv2.resize(current_roi, (self.reference_image.shape[1], self.reference_image.shape[0]))
        
        # Convert ROIs to grayscale
        current_gray = cv2.cvtColor(current_roi_resized, cv2.COLOR_BGR2GRAY)
        ref_gray = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2GRAY)
        
        # Compute absolute difference
        diff = cv2.absdiff(ref_gray, current_gray)
        
        # Apply threshold to highlight differences
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # Compute fault metrics
        total_diff_pixels = np.count_nonzero(thresh)
        total_pixels = thresh.shape[0] * thresh.shape[1]
        difference_percentage = (total_diff_pixels / total_pixels) * 100
        
        # Classify fault severity
        if difference_percentage < 1:
            fault_severity = "Intact"
        elif difference_percentage < 5:
            fault_severity = "Minor Fault"
        else:
            fault_severity = "Significant Fault"
        
        return {
            "difference_percentage": difference_percentage,
            "fault_severity": fault_severity,
            "total_diff_pixels": total_diff_pixels
        }
    
    def run(self):
        """
        Run real-time fault detection using webcam feed.
        """
        try:
            while True:
                # Capture frame-by-frame
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Store current frame for selection
                self.current_frame = frame.copy()
                display_frame = frame.copy()
                
                # Provide instructions
                if self.reference_image is None:
                    cv2.putText(display_frame, 
                                "Select Reference Image by clicking and dragging", 
                                (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, 
                                (0, 255, 0), 
                                2)
                elif self.roi is None:
                    cv2.putText(display_frame, 
                                "Select Detection ROI by clicking and dragging", 
                                (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, 
                                (0, 255, 0), 
                                2)
                
                # Draw reference ROI rectangle if selected
                if self.reference_roi:
                    x, y, w, h = self.reference_roi
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Draw current ROI rectangle if selected
                if self.roi:
                    x, y, w, h = self.roi
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Detect faults in ROI
                    fault_result = self.detect_faults(frame)
                    
                    # Determine text color based on fault severity
                    text_color = (0, 0, 255)  # Red for faults
                    if fault_result['fault_severity'] == "Intact":
                        text_color = (0, 255, 0)  # Green for intact
                    
                    # Display fault information
                    cv2.putText(display_frame, 
                                f"{fault_result['fault_severity']} "
                                f"({fault_result['difference_percentage']:.2f}%)", 
                                (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, 
                                text_color, 
                                2)
                
                # Display the frame
                cv2.imshow('Fault Detection', display_frame)
                
                # Break loop or continue
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # Press 'q' to quit
                    break
                
        finally:
            # Clean up
            self.cap.release()
            cv2.destroyAllWindows()

def main():
    try:
        # Initialize and run detector
        detector = WebcamFaultDetector()
        detector.run()
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()