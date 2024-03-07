from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)

from ast import literal_eval
import numpy as np
from glob import glob

def square_and_enlarge_bbox(bbox, factor=2.0, image_dims=None):
    """
    Squarify and enlarge a bounding box by a given factor.

    Parameters:
    - bbox (list): Input bounding box in the format [x1,y1,x2,y2].
    - factor (float): Enlargement factor.
    - image_dims (tuple, optional): Image dimensions as (width, height) to prevent the bbox from exceeding the boundaries.

    Returns:
    - list: Adjusted bounding box.
    """
    
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    
    # Calculate side length of squared bbox
    side_length = max(width, height)
    
    # Find center of original bbox
    cx, cy = x1 + width/2, y1 + height/2
    
    # Adjust coordinates to form a squared bbox
    half_side = side_length / 2
    x1, y1 = cx - half_side, cy - half_side
    x2, y2 = cx + half_side, cy + half_side
    
    # Enlarge the squared bbox by the given factor
    width = x2 - x1
    height = y2 - y1
    x1 -= (factor - 1) * width / 2
    y1 -= (factor - 1) * height / 2
    x2 += (factor - 1) * width / 2
    y2 += (factor - 1) * height / 2
    
    # If image dimensions are given, ensure the bbox does not exceed boundaries
    if image_dims:
        img_width, img_height = image_dims
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_width, x2)
        y2 = min(img_height, y2)
    
    return [x1, y1, x2, y2]

def bbox_transform(gt_box,
                   squarify=True,
                   scaling_factor=1.00, 
                   image_dims:tuple=(1920, 1280)):
    org_width, org_height = image_dims
    gt_box[0] = (gt_box[0])*org_width # width x1
    gt_box[2] = (gt_box[2])*org_width # width x2
    gt_box[1] = (gt_box[1])*org_height # height y1
    gt_box[3] = (gt_box[3])*org_height # height y2
    
    return square_and_enlarge_bbox(gt_box, scaling_factor, 
                                image_dims=(org_width, org_height))

import torch
import random
import torch.nn.functional as F

def vit_large(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

class dataset(Dataset):
    def __init__(self,
                 datasets:list=["waymo", "oxford"],
                 agents:list=[0,1,2,5,6,7],
                 split_type="train",
                 bbox_size="all", #small, large, all
                 resize_to = (224,224),
                 mean = [0.485, 0.456, 0.406], 
                 std = [0.229, 0.224, 0.225],
                 base_dir:str="annotations/",
                 anno_base_dir:str="annotations/files/",
                 annotation_file_path:str="road_labels.xlsx",
                 action_rec=False,
                 return_agent_frames=True,
                 return_full_video_frames=True,
                 bbox_scaling_factor=2.00):
        self.id2label = {'Move': 0,  'Brake': 1, 'Stop': 2, 'TurLft': 3, 'TurRht': 4, 'Cross': 5}
        self.label2id = {i:label for label,i in self.id2label.items()}
        self.action_rec = action_rec
        self.split = split_type
        self.included_datasets = datasets
        self.included_agents = [int(x) for x in agents]
        self.anno_base_dir = anno_base_dir
        self.num_classes = len(self.label2id)
        self.transforms = transforms.Compose([
                                transforms.Resize((224,224)),         
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std),])
        self.bbox_size = bbox_size
        self.df = self._load_df(annotation_file_path)
        self.dataset_list = list(range(len(self.df)))
        self.agent_label_dict = {0: 'Ped', 1: 'Car', 2: 'Cyc', 3: 'Mobike', 4: 'MedVeh', 5: 'LarVeh', 6: 'Bus',
 7: 'EmVeh',  8: 'TL',  9: 'OthTL'}
        self.label2id = {'Move': 0,  'Brake': 1, 'Stop': 2, 'TurLft': 3, 'TurRht': 4, 'Cross': 5}
        self.id2label = {i:label for label,i in self.id2label.items()}
        self.bbox_scaling_factor = bbox_scaling_factor
        self.num_classes = len(self.label2id)
        self.flip_transform = RandomHorizontalFlip(p=1.0)
        self.return_agent_frames = return_agent_frames
        self.return_full_video_frames = return_full_video_frames
    
    def _grouped_median_filter_large(self, group):
        median = group.median()
        return group[group >= median]
    def _grouped_median_filter_small(self, group):
        median = group.median()
        return group[group < median]

    def _load_df(self, file_path):
        df = pd.read_excel(file_path).drop(columns=["Unnamed: 0"])
        df = df[df["split"]==self.split]

        if self.bbox_size == "small":
            idx= df.groupby('agent_id')['avg_bbox'].apply(self._grouped_median_filter_small).reset_index()["level_1"].to_list()
            df = df.iloc[idx]
        elif self.bbox_size == "large":
            idx= df.groupby('agent_id')['avg_bbox'].apply(self._grouped_median_filter_large).reset_index()["level_1"].to_list()
            df = df.iloc[idx]
        if self.action_rec:
            df = df[df["action_labels"] !=-1] #filter unmapped action labels 
        # only include the specified dataset
        if len(self.included_datasets) ==1:
            df = df[df["ds"]==self.included_datasets[0]] 
        df["video-agent"] = df["video-agent"].apply(lambda x: literal_eval(x))
        # filter agents
        df = df[df["agent_id"].isin(self.included_agents)]

        return df.reset_index().drop(columns="index")


    def __len__(self):
        return len(self.dataset_list)
    
    def _load_indices(self, df):
        if self.split == "train":
            indices = torch.randperm(len(df)-1)[:16] #torch.randint(high=len(df_bbox), size=(16,))
            indices = indices.sort()[0]

        else:
            # Use np.linspace to generate evenly spaced indices
            indices = np.linspace(0, len(df)-1, 16)
            # Convert the indices to integers
            indices = np.round(indices).astype(int)
        return indices
    
    def _load_filenames(self, frames, 
                        video_path, dataset):
        if dataset =="waymo":
            frame_files = sorted([x for x in glob(f"./{dataset}/{video_path.split('_')[0]}/{video_path}/*")
                                             if x.split("/")[-1].replace(".jpg","") in frames])
        elif dataset == "oxford":
            frame_files = sorted([x for x in glob(f"./{dataset}/{video_path}/*") if x.split("/")[-1].replace(".jpg","") in frames])
        
        return frame_files
  
    def flip_video(self, video):
        return torch.flip(video, [3])
    
    def label_flip(self, label_tensor):
        label = label_tensor.argmax().item()
        
        # if label is turn left, make it turn right
        if label == 3:
            new_label = torch.tensor(4)
        # if label is turn right, make it turn left
        if label == 4:
            new_label = torch.tensor(3)
        # moving, forward etc, doesn't change the label
        else:
            new_label = label

        return F.one_hot(torch.tensor(new_label), self.num_classes).float()
        
    def horizontal_flip(self, data_dict):
        if random.random() > 0.5:   
            try: 
                data_dict["video"] = self.flip_video(data_dict["video"]) 
            except KeyError:
                pass
            
            try:
                data_dict["full_video"] = self.flip_video(data_dict["full_video"]) 
            except KeyError:
                pass
            data_dict["label"] = self.label_flip(data_dict["label"])

        return data_dict
    
    def _load_label(self, df):
        label = df["intention_labels"] if self.action_rec == False else df["action_labels"]
        one_hot_label = F.one_hot(torch.tensor(label), self.num_classes).float()
        return one_hot_label
        
    def __getitem__(self,index):
        data_dict = {}
        agent_df = self.df.loc[index]
        data_dict["label"] = self._load_label(agent_df)
        
        dataset = agent_df["ds"]
        image_dims = (1280,960) if dataset == "oxford" else (1920, 1280)
        video_path, agent = agent_df["video-agent"]
        # load the bboxes for a specific video
        df_bbox = pd.read_pickle(f"{self.anno_base_dir}{video_path}.p")
        
        df_bbox = df_bbox[(df_bbox["frame"]>= agent_df["min_frame"]) 
                       & (df_bbox["frame"]< agent_df["max_frame"]) 
                       & (df_bbox["agent"]== agent)].reset_index().drop(columns=["index"])
        
        indices = self._load_indices(df_bbox)
        df_bbox = df_bbox.loc[indices]       
        bbox_list = df_bbox.annotations.apply(lambda x: bbox_transform(x["box"],
                                                scaling_factor=self.bbox_scaling_factor,
                                                image_dims=image_dims)).tolist() 
        frames = [str(x).zfill(5) for x in df_bbox["frame"].tolist()]
        files = self._load_filenames(frames, video_path, dataset)
        
        if self.return_agent_frames:
            bbox_frames = torch.stack([self.transforms(default_loader(x).crop(bbox_list[i])) 
                                                    for i,x in enumerate(sorted(files))])

            data_dict["video"] = bbox_frames

        return data_dict
        
