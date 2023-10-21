# Self-Driving-Car

​In my pursuit to understand the basics of self-driving car technology, I embarked on this project that involved leveraging neural networks and data augmentation techniques. Here's an overview of the key parts of this project:

1. Leveraging Data:
I obtained road images and steering angle data, sourced from Udacity's Self-driving car simulator. This  dataset consisted of road images from three angles as well as other parameters such as steering angle, throttle, speed etc.  The road images and steering angle values were considered for a basic but robust self-driving model

2. Data Augmentation and Diversity:
To ensure the reliability and versatility of my model, I employed OpenCV and imgaug libraries to meticulously augment and diversify the dataset. This innovative approach enabled the generation of additional data, enhancing the model's performance and adaptability

3. Real-time Navigational:
The ultimate goal was to achieve a practically good navigational accuracy in self-driving scenarios. To achieve this, I designed and implemented a novel deep neural network architecture, inspired by NVIDIA's DAVE2 model. This architecture allowed me to map road images to precise steering angle values in real-time, enabling the self-driving car to navigate effectively and complete the route

