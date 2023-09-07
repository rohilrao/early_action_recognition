import cv2
import random
import numpy as np
import torch


class VideoCapture:
    @staticmethod
    def load_frames_from_video(video_path, percent=0.3):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        num_frames_to_read = min(15, max(1, int(total_frames * percent)))  # Ensure at least 1 frame

        if num_frames_to_read < 15:
            pad_frames = 15 - num_frames_to_read
        else:
            pad_frames = 0

        frames = []
        for _ in range(num_frames_to_read):
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                break

        cap.release()

        while pad_frames > 0:
            # Adding black frames at the start to pad
            frames.insert(0, np.zeros_like(frames[0]))
            pad_frames -= 1

        # Ensure exactly 15 frames
        while len(frames) < 15:
            frames.append(np.zeros_like(frames[0]))  # Adding black frames at the end
        
        
        # Crop or pad frames to 240 x 320
        cropped_or_padded_frames = []
        for frame in frames:
            h, w, _ = frame.shape
            if h < 224 or w < 224:
                # If frame is smaller, pad it
                pad_h = max(0, 224 - h)
                pad_w = max(0, 224 - w)
                frame = cv2.copyMakeBorder(frame, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            else:
                # If frame is larger, crop it
                frame = frame[:224, :224, :]
            cropped_or_padded_frames.append(frame)

        
        frames_array = np.stack(cropped_or_padded_frames, axis=0)   
        
        video_tensor = torch.tensor(frames_array, dtype=torch.float32) / 255.0  # Normalize to [0, 1]

        frame_idxs = list(range(0,15))

        return video_tensor, frame_idxs
    

'''
class VideoCapture:

    @staticmethod
    def load_frames_from_video(video_path,
                               num_frames,
                               sample='rand'):
        """
            video_path: str/os.path
            num_frames: int - number of frames to sample
            sample: 'rand' | 'uniform' how to sample
            returns: frames: torch.tensor of stacked sampled video frames 
                             of dim (num_frames, C, H, W)
                     idxs: list(int) indices of where the frames where sampled
        """
        cap = cv2.VideoCapture(video_path)
        assert (cap.isOpened()), video_path
        vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # get indexes of sampled frames
        acc_samples = min(num_frames, vlen)
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []

        # ranges constructs equal spaced intervals (start, end)
        # we can either choose a random image in the interval with 'rand'
        # or choose the middle frame with 'uniform'
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
        else:  # sample == 'uniform':
            frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]

        frames = []
        for index in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, frame = cap.read()
            if not ret:
                n_tries = 5
                for _ in range(n_tries):
                    ret, frame = cap.read()
                    if ret:
                        break
            if ret:
                #cv2.imwrite(f'images/{index}.jpg', frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame)
                # (H x W x C) to (C x H x W)
                frame = frame.permute(2, 0, 1)
                frames.append(frame)
            else:
                raise ValueError

        while len(frames) < num_frames:
            frames.append(frames[-1].clone())
            
        frames = torch.stack(frames).float() / 255
        cap.release()
        return frames, frame_idxs
'''