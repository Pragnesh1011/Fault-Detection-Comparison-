import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_image_faults(image1_path, image2_path):
    """
    Detect faults between two images of the same artifact.
    
    Parameters:
    image1_path (str): Path to the first image (original/intact)
    image2_path (str): Path to the second image (potentially damaged)
    
    Returns:
    dict: A dictionary containing comparison results and visualization
    """
    # Read images
    img1 = cv2.imread("D:\WhatsApp_Image_2024-12-14_at_01.52.41_17eb0c9f-removebg.png")
    img2 = cv2.imread("D:\WhatsApp_Image_2024-12-14_at_01.52.49_42eaf871-removebg-preview.png")
    
    # Validate input images
    if img1 is None or img2 is None:
        raise ValueError("Unable to read one or both images. Check file paths.")
    
    # Resize images to same dimensions if needed
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
    
    # Convert to grayscale for comparison
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Compute absolute difference
    diff = cv2.absdiff(gray1, gray2)
    
    # Apply threshold to highlight differences
    thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)[1]
    
    # Find contours of differences
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Compute fault metrics
    total_diff_pixels = np.sum(thresh == 255)
    difference_percentage = (total_diff_pixels / (thresh.shape[0] * thresh.shape[1])) * 100
    
    # Visualize differences
    img_diff_viz = img2.copy()
    cv2.drawContours(img_diff_viz, contours, -1, (0, 0, 255), 2)
    
    # Classify fault severity
    if difference_percentage < 1:
        fault_severity = "Minor"
    elif difference_percentage < 5:
        fault_severity = "Moderate"
    else:
        fault_severity = "Significant"
    
    return {
        "difference_percentage": difference_percentage,
        "fault_severity": fault_severity,
        "total_diff_pixels": total_diff_pixels,
        "difference_image": thresh,
        "annotated_image": img_diff_viz,
        "contours": contours
    }

def visualize_fault_detection(result):
    """
    Create a visualization of the fault detection results.
    
    Parameters:
    result (dict): Results from detect_image_faults function
    """
    plt.figure(figsize=(15, 5))
    
    # Original Difference Image
    plt.subplot(131)
    plt.title("Difference Heatmap")
    plt.imshow(result['difference_image'], cmap='hot')
    plt.axis('off')
    
    # Annotated Image
    plt.subplot(132)
    plt.title("Annotated Image")
    # Convert BGR to RGB for correct color display
    annotated_img_rgb = cv2.cvtColor(result['annotated_image'], cv2.COLOR_BGR2RGB)
    plt.imshow(annotated_img_rgb)
    plt.axis('off')
    
    # Textual Results
    plt.subplot(133)
    plt.title("Fault Analysis")
    plt.text(0.5, 0.7, f"Difference: {result['difference_percentage']:.2f}%", 
             horizontalalignment='center')
    plt.text(0.5, 0.5, f"Severity: {result['fault_severity']}", 
             horizontalalignment='center')
    plt.text(0.5, 0.3, f"Diff Pixels: {result['total_diff_pixels']}", 
             horizontalalignment='center')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage
def main():
    try:
        # Replace these with actual image paths
        image1_path = 'original_artifact.jpg'
        image2_path = 'potentially_damaged_artifact.jpg'
        
        # Detect faults
        fault_result = detect_image_faults(image1_path, image2_path)
        
        # Visualize results
        visualize_fault_detection(fault_result)
        
        print(f"Fault Severity: {fault_result['fault_severity']}")
        print(f"Difference Percentage: {fault_result['difference_percentage']:.2f}%")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()