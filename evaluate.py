import random
import torch

from scripts.demo.turbo import *
from sgm.modules.diffusionmodules.openaimodel import get_feature_dic
from mmdet.apis import inference_detector
from utils import IoU, get_rand


def evaluate(pretrain_detector, seg_module, diffusion_model, 
             class_seen, class_unseen, exp_dir:str, 
             eval_iter:int = 1000, H:int = 512, W:int = 512):
    
    model, sampler, state = diffusion_model

    print('***********************   begin   **********************************')
    print(f"Start evaluate with maximum {eval_iter} iterations.")

    batch_size = 1
    
    with torch.no_grad():
        for v, classes in enumerate([class_seen, class_unseen]):
            iou = 0
            for j in range(eval_iter):
                
                print('Iter ' + str(j) + '/' + str(eval_iter))
                trainclass = classes[random.randint(0, len(classes)-1)]
                prompt = "a photograph of a " + trainclass
                print(f"Iter {j}: prompt--{prompt}")
                assert prompt is not None
                data = [batch_size * [prompt]]
                        
                for prompts in data:
                    
                    # generate images
                    seed = get_rand()
                    out = sample(
                        model, sampler, H=H, W=W, seed=seed, 
                        prompt=prompts[0], filter=state.get("filter")
                    )
                    x_sample_list = [out[0]]
                    
                    # detector
                    result = inference_detector(pretrain_detector, x_sample_list)
                    seg_result_list = []
                    flag = False # detect if mmdet fail to detect the object
                    for i in range(len(result)):
                        seg_result = result[i].pred_instances.masks
                        if len(seg_result) > 0: 
                            flag = False
                            seg_result_list.append(seg_result[0].unsqueeze(0))
                        else: 
                            flag = True
                            break
                    if flag:
                        print("pretrain detector fail to detect the object in the class:", trainclass) 
                        continue
                    
                    # get class embedding
                    class_embedding, uc = sample(
                        model, sampler, condition_only=True, H=H, W=W, seed=seed, 
                        prompt=trainclass, filter=state.get("filter")
                    )
                    class_embedding = class_embedding['crossattn']
                    if class_embedding.size()[1] > 1:
                        class_embedding = torch.unsqueeze(class_embedding.mean(1), 1)
                    class_embedding = class_embedding.repeat(batch_size, 1, 1)
                    
                    # seg_module
                    diffusion_features = get_feature_dic()
                    total_pred_seg = seg_module(diffusion_features, class_embedding)
                    
                    for b_index in range(batch_size):
                        pred_seg = total_pred_seg[b_index]

                        label_pred_prob = torch.sigmoid(pred_seg)
                        label_pred_mask = torch.zeros_like(label_pred_prob, dtype=torch.float32)
                        label_pred_mask[label_pred_prob > 0.5] = 1
                        annotation_pred = label_pred_mask.cpu()

                        if len(seg_result_list[b_index]) == 0:
                            print("pretrain detector fail to detect the object in the class:", trainclass)
                        else:
                            seg = seg_result_list[b_index]
                            annotation_pred_gt = seg.float().cpu()
                            iou += IoU(annotation_pred_gt, annotation_pred)

            with open(f'{exp_dir}/ious_{v}.txt', "w") as f:
                f.write(str(iou/eval_iter)+'\n')
                