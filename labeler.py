import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Global variables
points = []
class_id = 0
initial_point = None
click_tolerance = 10
def onclick(event):
    global points, initial_point

    # Capture x, y coordinates of the click
    x, y = event.xdata, event.ydata
    if x is not None and y is not None:
        if len(points) == 0:
            initial_point = (x, y)
        points.append((x, y))

        # Plot the point on the image
        plt.plot(x, y, 'ro')

        if len(points) > 1:
            plt.plot([points[-2][0], points[-1][0]], [points[-2][1], points[-1][1]], 'r-')  # red line
        
        if len(points) > 2:
            distance = np.sqrt((x - initial_point[0])**2 + (y - initial_point[1])**2)
            if distance <= click_tolerance:
                plt.plot([points[-1][0], points[0][0]], [points[-1][1], points[0][1]], 'r-')  # Close the polygon
                print('Polygon closed.')
                plt.draw()
                save_annotation(image_path, points)
                plt.disconnect(cid)
                return

        plt.draw()

def save_annotation(image_path, points):
    # Normalize points
    img = Image.open(image_path)
    img_width, img_height = img.size
    normalized_points = [(p[0] / img_width, p[1] / img_height) for p in points]

    annotation = f'{class_id} '
    for p in normalized_points:
        annotation += f'{p[0]} {p[1]} '

    # Save
    annotation_filename = image_path.split('/')[-1].split('.')[0] + '.txt'
    with open(annotation_filename, 'w') as f:
        f.write(annotation.strip())
    
    print(f'Annotation saved to {annotation_filename}')

# Main function to run the program
def create_annotation(image_path):
    global points
    
    # Loadings
    img = Image.open(image_path)
    img_width, img_height = img.size
    fig, ax = plt.subplots()
    ax.imshow(img)

    # Connect all events
    global cid
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    print('Click on the image to create segmentation points. Click again on the first point (within tolerance) to close the polygon and save the annotation.')

    plt.show()

image_path = 'IM-0003-0030.png'  # Replace with your image path
create_annotation(image_path)
