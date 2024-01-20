import os

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
class_dict = {}
coco_class_seq = 'person, bicycle, car, motorbike, bus, truck, boat, traffic light, fire hydrant, stop sign, bench, \
    bird, dog, horse, sheep, cow, elephant, zebra, giraffe, backpack, umbrella, handbag, tie, skis, \
    sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, \
    cup, knife, spoon, bowl, banana, apple, orange, broccoli, carrot, pizza, donut, cake, \
    chair, pottedplant, bed, diningtable, tvmonitor, laptop, remote, keyboard, cell phone, \
    microwave, oven, sink, refrigerator, book, clock, vase, scissors, teddy bear, toothbrush, \
    aeroplane, train, parking meter, cat, bear, suitcase, frisbee, snowboard, fork, sandwich, hot dog, \
    toilet, toaster, hair drier'

split_seen_class = ' laptop, remote, keyboard, cell phone, microwave, oven, sink, refrigerator, book, clock, vase,\
scissors, teddy bear, toothbrush, aeroplane, train, parking meter, cat, bear, suitcase, frisbee,\
snowboard, fork, sandwich, hot dog, toilet, toaster, hair drier, person, bicycle, car, motorbike,\
bus, truck, boat, traffic light, fire hydrant, stop sign, bird, dog, horse, sheep, cow,\
elephant, zebra, giraffe, backpack, umbrella, handbag, tie, skis, sports ball, kite, baseball bat,\
baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, knife, spoon, bowl'

split_unseen_class = 'banana, apple, orange,\
broccoli, carrot, pizza,\
donut, cake, chair, bench,\
pottedplant, bed, diningtable, tvmonitor'

def split(path: str, dataset: str):
    if dataset == 'coco':
        class_dict = {category.strip(): idx for idx, category in enumerate(coco_class_seq.split(','))}
        print(class_dict)
        assert len(class_dict.keys()) == 78 # without mouse and remove duplicated bench
    
    with open(path, 'w') as f:
        for seen_class in split_seen_class.split(','):
            seen_class = seen_class.strip()
            f.write(f'{seen_class},{class_dict[seen_class]}\n')
        for unseen_class in split_unseen_class.split(','):
            unseen_class = unseen_class.strip()
            f.write(f'{unseen_class},{class_dict[unseen_class]}\n')
            
if __name__ == '__main__':
    file_id = 6
    path = f'{current_directory}/class_split{file_id}.csv'
    split(path, 'coco')