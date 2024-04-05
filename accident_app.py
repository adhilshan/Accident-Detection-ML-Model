import os
import cv2
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.models import load_model
import subprocess
import shutil
from multiprocessing import Pool
import tempfile
import streamlit as st
import base64
import subprocess
if not os.path.isfile('model.h5'):
    subprocess.run(['curl --output model.h5 "https://github.com/adhilshan/Accident-Detection-ML-Model/raw/main/Updated_80_percent_new_model.h5"'], shell=True)

# [All the functions from first code option]

def load_and_preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = crop_center_square(frame)
        frame = cv2.resize(frame, (224, 224))
        frame = frame[:, :, [2, 1, 0]]
        frames.append(frame)
    cap.release()

    return np.array(frames)


def calculate_optical_flow(frames):
    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
    optical_flow_frames = []
    for i in range(len(gray_frames) - 1):
        flow = cv2.calcOpticalFlowFarneback(gray_frames[i], gray_frames[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        optical_flow_frames.append(flow)
    optical_flow_frames.append(optical_flow_frames[-1])
    return np.array(optical_flow_frames)



def parallel_optical_flow(chunks):
    with Pool(processes=os.cpu_count()) as pool:
        optical_flows = pool.map(calculate_optical_flow, chunks)
    return optical_flows


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = x // 2 - min_dim // 2
    start_y = y // 2 - min_dim // 2
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]
def load_and_preprocess_video_every_5th_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 5 == 0:
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, (224, 224))
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)
        frame_count += 1
    cap.release()
    return np.array(frames)



def pad_chunk(chunk, window_size=30):
    while chunk.shape[0] < window_size:
        chunk = np.vstack((chunk, [chunk[-1]]))  # appending the last frame to the chunk
    return chunk

def create_chunks_from_frames(frames, window_size=30):
    # Create non-overlapping chunks of window_size from frames
    chunks = [frames[i:i+window_size] for i in range(0, len(frames), window_size)]
    if len(chunks[-1]) < window_size:
        chunks[-1] = pad_chunk(chunks[-1])
    return chunks

def overlay_predictions_to_video(frames, predictions):
    temp_dir = 'temp_frames'
    
    # Clear existing frames and video if they exist
    if os.path.exists(temp_dir):
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    video_output_path = 'p_user_upload.mp4'
    if os.path.exists(video_output_path):
        os.remove(video_output_path)

    frame_idx = 0

    # Desired resolution for the video
    desired_resolution = (1280, 720)  # HD resolution

    for prediction in predictions:
        # Overlay the prediction for WINDOW_SIZE frames
        if frame_idx >= len(frames):  # Make sure not to exceed total frames
            break
        frame = frames[frame_idx]
        
        # Resize the frame to the desired resolution
        frame = cv2.resize(frame, desired_resolution, interpolation=cv2.INTER_AREA)
        
        frame_idx += 1
        color = (0, 255, 0) if prediction[1] > prediction[0] else (255, 0, 0)
        frame = cv2.putText(frame, f"Accident: {prediction[0]:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        frame = cv2.putText(frame, f"No Accident: {prediction[1]:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        frame = frame[:, :, [2, 1, 0]]
            
        # Save frame to disk
        cv2.imwrite(os.path.join(temp_dir, f'frame_{frame_idx:04d}.png'), frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])  # Highest quality

    # Use ffmpeg to stitch frames into video with higher bitrate for better quality
    cmd = f"ffmpeg -framerate 20 -i {temp_dir}/frame_%04d.png -c:v libx264 -b:v 1500k -pix_fmt yuv420p {video_output_path}"
    subprocess.call(cmd, shell=True)

    
def process_video(video_path, model):
    # Load all frames
    frames = load_and_preprocess_video(video_path)
    
    # Create chunks of size 30 from frames
    chunks = create_chunks_from_frames(frames)

    # Calculate optical flow for all chunks
    optical_flows = parallel_optical_flow(chunks)
    optical_flows = [flow / np.max(np.abs(flow), axis=(1, 2), keepdims=True) for flow in optical_flows]
    
    # Normalize frames
    chunks = [chunk / 255.0 for chunk in chunks]

    # Batch predictions
    all_predictions = []
    
    for i in range(len(chunks)):
        batched_frames = np.array([chunks[i]])
        batched_flows = np.array([optical_flows[i]])
        prediction = model.predict([batched_frames, batched_flows])
        #print(prediction)
        all_predictions.extend([prediction[0]] * WINDOW_SIZE)

    # Overlay predictions to the video and save
    overlay_predictions_to_video(frames, all_predictions)
    #return all_predictions    
#___________________________________________________________________________________________________

# [All the functions from the second set of code]
def second_calculate_optical_flow(frames):
    gray_frames = [cv2.cvtColor(tf.cast(frame, tf.uint8).numpy(), cv2.COLOR_RGB2GRAY) for frame in frames]
    optical_flow_frames = []
    for i in range(len(gray_frames) - 1):
        flow = cv2.calcOpticalFlowFarneback(gray_frames[i], gray_frames[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        optical_flow_frames.append(flow)

    # Repeat the last optical flow frame
    optical_flow_frames.append(optical_flow_frames[-1])
    #optical_flow_frames
    optical_flow_frames=np.array(optical_flow_frames)
    return optical_flow_frames

def singledatacombined_load_and_preprocess_video(video_path,max_frames=30):

    cap = cv2.VideoCapture(video_path)
    frames = np.zeros(shape=(max_frames, 224, 224, 3))
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    i = 0
    frame_count = 0
    try:
        while True:
            (ret, frame) = cap.read()
            if not ret:
                break
            if frame_count %5  == 0:
                frame = crop_center_square(frame)
                frame = cv2.resize(frame, (224, 224))
                frame = frame[:, :, [2, 1, 0]]
                frames[i] = frame
                i += 1
                if i == max_frames: 
                    break
            frame_count += 1
    finally:
        cap.release()
        st.write("Total Frames:", total_frames)
        #print("Frames per Second:", fps)
        st.write("frame processed:",len(frames))

    return tf.constant(frames,dtype=tf.float32)#, Label #(tf.constant(frames, dtype=tf.float32))/ 255.0, Label


def singlegenerator(video_path, max_frames=30,augment_data=False):
    frames = singledatacombined_load_and_preprocess_video(video_path)
    optical=second_calculate_optical_flow(frames)
    optical_flow = tf.convert_to_tensor(optical)

    optical_flow = optical_flow / tf.reduce_max(tf.abs(optical_flow)) #Normalize oprical flow
    if augment_data:
        # Apply data augmentation to frames
        augmented_frames = []
        random_num = random.random()
        for frame in frames:
            if random_num < 0.25:
                augmented_frame = tf.image.random_flip_left_right(frame)
            elif random_num < 0.5:
                augmented_frame = tf.image.random_flip_up_down(frame)
            elif random_num < 0.75:
                num_rotations = random.randint(0, 3)
                augmented_frame =tf.image.rot90(frame, k=num_rotations)
            else:
                augmented_frame = frame
            augmented_frames.append(augmented_frame)
        frames = tf.stack(augmented_frames)#np.array(augmented_frames)
    frames=frames/ 255.0
    return (frames, optical_flow)#, label

def single_video_predict_on_frames(vid_dir):
    st.write("===================================")
    frames, optical_flow = singlegenerator(vid_dir,augment_data=False)
    # Model prediction
    prediction = loaded_model([frames[tf.newaxis, ...], optical_flow[tf.newaxis, ...]])
    labels_map = ["Accident", "No Accident"]
    video_name=str(vid_dir)#.split('/')[-1])
    # Extracting max prediction and its index
    max_index = tf.argmax(prediction[0]).numpy()
    max_value = prediction[0][max_index].numpy()
   
    st.write(f"Name: Uploaded Video")
    st.write(f"Action Detected: {labels_map[max_index]} ({max_value*100:.2f}%)")
    #st.write("-----------------------------------")
    st.write(f"{labels_map[0]} Probability: {prediction[0][0]*100:.2f}%")
    st.write(f"{labels_map[1]} Probability: {prediction[0][1]*100:.2f}%")
    st.write("===================================")
    return prediction#[0]

# Global Constants
WINDOW_SIZE = 30
SAMPLE_VIDEOS_UNTRIMMED = ["Video4.mp4", "Video5.mp4", "Video6.mp4","Video7.mp4","Video8.mp4","Video9.mp4", "Video10.mp4"]
SAMPLE_VIDEOS_TRIMMED = ["Video1.mp4", "Video2.mp4", "Video3.mp4"]

# Ensure your model is loaded globally
loaded_model = load_model('model.h5')

def display_selected_sample_video(videos_list):
    selected_video = st.selectbox("Select a sample video to play:", videos_list)
    if os.path.exists(selected_video):
        st.video(selected_video)
    return selected_video
   

def get_image_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

    
def main():
    # Page Settings
    st.set_page_config(
        page_title="Accident Detection Model",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
            /* Styles for entire page and main container */
            body {
                background-color: #e6e6e6;
            }
            .stApp {
                background-color: #e6e6e6;
            }

            /* Styles for the header container */
            .header-container {
                background-color: #3a8d8b;  /* Adjust the color if needed */
                padding: 20px 40px;  /* Adjusted padding to push it away from the edges a bit */
                border-radius: 0;    /* Remove the rounded corners */
                margin: -10px -40px 10px -40px;   /* Stretching the header to the full width */
            }

            /* Styling the text color inside the header */
            .header-container h1, .header-container h2 {
                color: white;
            }

        </style>
        <div class="header-container">
            <h1>Accident Detection Model</h1>
            <h2>Dissertation on Accident Detection for Smart City Transportation</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """Developer: Adhil Shan | 
        [Research Paper](https://arxiv.org/pdf/2310.10038.pdf)""",
        unsafe_allow_html=True
    )

    st.warning("The models are still in development and were originally trained to detect trimmed 5 seconds non-overlapping actions.")

    
    video_option = st.radio("", ["Untrimmed (Accident Detection)", "Trimmed (5 Seconds window)"])
    st.markdown("<div class='big-heading'>Upload your own video or use any of the sample videos below:</div>", unsafe_allow_html=True)

    # This makes the upload button appear at the top
    uploaded_file = st.file_uploader("", type=['mp4', 'mov', 'avi', 'mkv'])

    if video_option == "Untrimmed (Accident Detection)":
        st.markdown("## Sample Videos:")
        col1, col2 = st.columns(2)
        with col1: 
            display_selected_sample_video(SAMPLE_VIDEOS_UNTRIMMED)

        #uploaded_file = st.file_uploader("Upload own video:", type=['mp4', 'mov', 'avi', 'mkv'])
        with col2:
            if uploaded_file:
                tfile = tempfile.NamedTemporaryFile(delete=False) 
                tfile.write(uploaded_file.read())
                #newcol1, newcol2 = st.columns(2)  # Splitting the layout

                progress_bar = st.progress(0)
                st.write('Processing video...')
                process_video(tfile.name, loaded_model)  # Assuming your function for processing untrimmed videos

                progress_bar.progress(50)

                if os.path.exists('p_user_upload.mp4'):
                    st.write('Video processed. Displaying results...')
                    #st.video('p_user_upload.mp4')

                    st.video('p_user_upload.mp4')
                    progress_bar.progress(100)
                else:
                    st.write("Error: Video processing failed.")

                os.remove(tfile.name)
    
    elif video_option == "Trimmed (5 Seconds window)":
        st.markdown("## Sample Videos:")
        col1, col2 = st.columns(2)
        with col1: 
            selected_video_file = display_selected_sample_video(SAMPLE_VIDEOS_TRIMMED)
        with col2: 
            progress_bar = st.progress(0)
            st.write('Processing video...')
            single_video_predict_on_frames(selected_video_file)
            progress_bar.progress(100)
        
        #uploaded_file = st.file_uploader("Upload your own video:", type=['mp4', 'mov', 'avi', 'mkv'])
        
        if uploaded_file:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())
            
            st.write('Displaying uploaded video...')
            #st.video(tfile.name)
            col1, col2 = st.columns(2)  # Splitting the layout
            col1.video(tfile.name)
            with col2: 
                progress_bar = st.progress(0)
                st.write('Processing video...')
                single_video_predict_on_frames(tfile.name)  # Assuming your function for processing trimmed videos
                progress_bar.progress(100)
            
            os.remove(tfile.name)

if __name__ == "__main__":
    main()
