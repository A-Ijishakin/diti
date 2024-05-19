
class ImgLoader(torch.utils.data.Dataset):
    def __init__(self, path='', starting_index=0, h_creation=False, ending_index=-1, 
                 selected_indices=False, 
                 device_num=0):
        imgs = f'{path}/*'  
        
        self.image_list = sorted(glob(imgs))[starting_index:ending_index] 
        
        if selected_indices:
            loaded = []
            
            for latent in glob('/home/rmapaij/HSpace-SAEs/datasets/CELEB-A/h_seven/*'):
                loaded.append(latent) 
            
            loaded = [f'{path}/{os.path.basename(latent).replace(".pt", ".jpg")}' for latent in loaded]     
            
            print('Loaded num:', len(loaded)) 
            
            for img in tqdm(loaded):
                if img in self.image_list:
                    self.image_list.remove(img)
                    
            print('Length after Pruning:', len(self.image_list)) 
        
        self.h_creation = h_creation 
        
        self.transform = tfs.Compose([
            tfs.ToTensor(), 
            tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        ])
        
        self.device_num = device_num 
        
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx, native=True, xs=None): 
        x = Image.open(self.image_list[idx]) if native else Image.open(xs[idx])
        
        # Crop the center of the image
        w, h = x.size 
        crop_size = min(w, h)

        left    = (w - crop_size)/2
        top     = (h - crop_size)/2
        right   = (w + crop_size)/2
        bottom  = (h + crop_size)/2

        # Crop the center of the image
        x = x.crop((left, top, right, bottom))

        # resize the image
        x = x.resize((512, 512))      
                
        if self.transform is not None:
            x = self.transform(x) 
            
        path = self.image_list[idx] if self.h_creation else ''
        
        
        return {'img': x.to(device=f'cuda:{self.device_num}', dtype=torch.float32), 
                'index' : idx, 'path': path}   
        