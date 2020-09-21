import os
import io
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from deploying_pytorch.pytorch_flask_service.model import MobileNetV2
from f_tools.GLOBAL_LOG import flog
from object_detection.f_fit_tools import sysconfig

app = Flask(__name__)
CORS(app)  # 解决跨域问题


def transform_image(image_bytes):
    '''
    字节图片流预处理
    :param image_bytes: 图处的字符流
    :return:
    '''
    my_transforms = transforms.Compose([
        transforms.Resize(255),  # 图片比例不变
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes))  # 读取2进制数据

    if image.mode != "RGB":
        __s = "input file does not RGB image..."
        flog.error(' %s', __s)
        raise ValueError(__s)
    # 前面增加一维
    __img = my_transforms(image)
    # 显示测试
    # import matplotlib.pyplot as plt
    # plt.imshow(image)
    # plt.show()
    return __img.unsqueeze(0).to(device)


def get_prediction(image_bytes):
    try:
        tensor = transform_image(image_bytes=image_bytes)
        flog.debug('预处理完成 %s', tensor.shape)

        # 正向推理
        __outs = model.forward(tensor).squeeze()

        flog.debug('正向完成 %s, %s', __outs.shape, __outs)
        outputs = torch.softmax(__outs, dim=0)
        prediction = outputs.detach().cpu().numpy()

        flog.debug('输出 %s', prediction)
        template = "class:{:<15} probability:{:.3f}"
        index_pre = [(class_indices[str(index)], float(p)) for index, p in enumerate(prediction)]
        # sort probability
        index_pre.sort(key=lambda x: x[1], reverse=True)
        text = [template.format(k, v) for k, v in index_pre]
        return_info = {"result": text}
    except Exception as e:
        return_info = {"result": [str(e)]}
    return return_info


@app.route("/predict", methods=["POST"])
@torch.no_grad()
def predict():
    image = request.files["file"]
    img_bytes = image.read()
    info = get_prediction(image_bytes=img_bytes)
    return jsonify(info)


@app.route("/", methods=["GET", "POST"])
def root():
    return render_template("up.html")


if __name__ == '__main__':
    '''------------------系统配置---------------------'''
    PATH_WEIGHTS = "./MobileNetV2(flower).pth"
    PATH_CLASS_JSON = "./class_indices.json"

    # assert os.path.exists(PATH_WEIGHTS), "weights path does not exist..."
    assert os.path.exists(PATH_CLASS_JSON), "class json path does not exist..."

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'  # 写死cpu
    # flog.info('输入索引，返回gpu名字 %s', torch.cuda.get_device_name(0))
    flog.info(device)

    '''---------------数据加载及处理--------------'''
    # load class info
    json_file = open(PATH_CLASS_JSON, 'rb')
    class_indices = json.load(json_file)

    '''------------------模型定义---------------------'''
    model = MobileNetV2(num_classes=5)
    if os.path.exists(PATH_WEIGHTS):
        model.load_state_dict(torch.load(PATH_WEIGHTS, map_location=device))
    model.to(device)
    model.eval()

    app.run(host="0.0.0.0", port=5000)
