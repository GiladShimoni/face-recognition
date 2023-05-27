import cv2


def detect_faces(image_path):
    # Load the Haar cascade XML file for face detection
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                    'haarcascade_frontalface_default.xml')
    print("Loading CascadeClassifier XML file for face detection")
    cv2.waitKey(0)


    
    # Load the input image
    image = cv2.imread(image_path)
    cv2.imshow('Input Image', image)
    cv2.waitKey(0)

    
    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray', gray)
    

    # Perform face detection
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))




    # Iterate over detected faces and output cropped images
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Crop the face region from the image
        face = image[y:y+h, x:x+w]

        # Display the image at each stage of the detection process
        cv2.imshow('Face Detection', image)
        cv2.waitKey(0)

        # Display the cropped face image
        cv2.imshow('Detected Face', face)
        cv2.waitKey(0)

    # Close all windows
    cv2.destroyAllWindows()

# Specify the path to your input image
image_path = 'test_image.jpg'

# Perform face detection and display the results
detect_faces(image_path)
