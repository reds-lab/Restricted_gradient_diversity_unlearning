import torch
import dnnlib
from torchvision import transforms

class Predictor:
  def __init__(self, mode='dla', device=torch.device('cuda')):
    self.mode = mode
    self.device = device
    if mode == 'dla':
      net = dnnlib.SimpleDLA()
      net = net.to(device)
      net = net.eval()
    
      checkpoint = torch.load(model_path)
      state_dict = dict([('.'.join(k.split('.')[1:]), checkpoint['net'][k]) for k in checkpoint['net']])
      net.load_state_dict(state_dict)

      self.transform = transforms.Compose([
          lambda x: (x.float() / 255.).clip(min=0., max=1.),
          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
      ])
    else:
      from transformers import CLIPProcessor, CLIPModel
      
      self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
      self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
      # classes = ('plane', 'car', 'bird', 'cat',
      #      'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
      classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck', 'random noise')
      self.class_tokens = self.processor(
          text=[f"a photo of a {x}" for x in classes], return_tensors="pt", padding=True)
      self.transform = lambda x: self.processor(images=x, return_tensors="pt")['pixel_values']

  def predict(self, images):
    if self.mode == 'dla':
      assert False
      return self.net(images)
    elif self.mode == 'clip':
      inputs = {
          'input_ids': self.class_tokens['input_ids'].to(images.device),
          'attention_mask': self.class_tokens['attention_mask'].to(images.device),
          'pixel_values': images
      }
      outputs = self.model(**inputs)
      logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
      return logits_per_image