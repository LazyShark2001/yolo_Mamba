import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import os

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_json', type=str, default=r'cocodataset/annotations/instances_val2017.json', help='training model path')
    parser.add_argument('--pred_json', type=str, default='runs/detect/val4/predictions.json', help='data yaml path')
    parser.add_argument('--endswith', type=str, default='.jpg', help='swith')

    return parser.parse_known_args()[0]

if __name__ == '__main__':
    opt = parse_opt()
    anno_json = opt.anno_json
    pred_json = opt.pred_json
    endswith = opt.endswith
    # 读取源文件
    with open(anno_json, 'r') as file:
        data_anno = json.load(file)

    with open(pred_json, 'r') as file:
        data_pred = json.load(file)

    dic = {}
    for item in data_anno['images']:
        dic[item['file_name']] = item

    # 对每个字典进行处理
    for item in data_pred:
        # 如果字典中包含 'image_id' 键
        if 'image_id' in item:
            # 将 'image_id' 值转换为5位数形式，并在前面用0填充
            try:
                item['image_id'] = dic[item['image_id'] + endswith]['id']  # 注意尾缀
            except:
                print(item['image_id'])

    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建新文件的路径
    new_file_path = os.path.join(current_dir, 'cache_coco.json')

    with open(new_file_path, 'w') as file:
        json.dump(data_pred, file, indent=4)

    anno = COCO(anno_json)  # init annotations api
    pred = anno.loadRes(new_file_path)  # init predictions api
    eval = COCOeval(anno, pred, 'bbox')
    eval.evaluate()
    eval.accumulate()
    eval.summarize()

    os.remove(new_file_path)
