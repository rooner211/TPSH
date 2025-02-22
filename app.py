from flask import Flask, render_template, request
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import os
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

MODELS = {
    'mobilenet': models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2),
    'resnet': models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2),
    'vgg': models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
}

imagenet_labels = models.MobileNet_V2_Weights.IMAGENET1K_V2.meta["categories"]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, model_name):
    model = MODELS[model_name]
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    return preprocess(img).unsqueeze(0)

def predict(image_tensor, model_name):
    model = MODELS[model_name]
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    top5_probs, top5_ids = torch.topk(probabilities, 5)
    return [(imagenet_labels[idx.item()], prob.item()) for prob, idx in zip(top5_probs, top5_ids)]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        model_name = request.form.get('model', 'mobilenet')
        if model_name not in MODELS:
            return render_template('index.html', error="Выбранная модель не поддерживается")

        file = request.files.get('file')
        if not file or not allowed_file(file.filename):
            return render_template('index.html', error="Некорректный файл")

        filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            image_tensor = preprocess_image(file_path, model_name)
            predictions = predict(image_tensor, model_name)
            return render_template('index.html', predictions=predictions, image_url=file_path, model=model_name)
        except Exception as e:
            return render_template('index.html', error=f"Ошибка обработки: {e}")
    
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)