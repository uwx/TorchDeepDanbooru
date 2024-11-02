from io import BytesIO
import os
from typing import IO, Generator, Optional
from PIL import Image
import numpy as np
import torch
import tqdm

from flask import Flask, flash, request, redirect, json
from werkzeug.utils import secure_filename

import deep_danbooru_model

model = deep_danbooru_model.DeepDanbooruModel()
model.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model-resnet_custom_v3.pt')))

model.eval()
model.half()
model.cuda()

def evaluate(fp: IO) -> Generator[tuple[str, float], None, None]:
    pic = Image.open(fp).convert("RGB").resize((512, 512))
    a = np.expand_dims(np.array(pic, dtype=np.float32), 0) / 255

    with torch.no_grad(), torch.autocast("cuda"):
        x = torch.from_numpy(a).cuda()

        # first run
        y: list[float] = model(x)[0].detach().cpu().numpy()

        # measure performance
        for n in tqdm.tqdm(range(10)):
            model(x)


    for i, p in enumerate(y):
        #if p >= 0.5:
        yield (model.tags[i], p)

def run_server(threshold: float, host: str, port: int, cpu: bool, auth_key: Optional[str]):
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    app = Flask("hydrus-dd lookup server")
    ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

    def allowed_file(filename: str | None):
        return filename is not None and '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    @app.route('/', methods=['GET', 'POST'])
    def upload_file():
        if request.method == 'POST':
            if auth_key:
                if request.headers.get('authorization') != f'Bearer {auth_key}':
                    response = app.response_class(
                        response=json.dumps({'error': 'invalid auth key'}),
                        status=401,
                        mimetype='application/json'
                    )
                    return response

            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)

            file = request.files['file']
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)

            if file and allowed_file(file.filename):
                _ = secure_filename(file.filename)
                image_path = BytesIO(file.read())
                results = [*((tag, float(p)) for (tag, p) in evaluate(image_path) if p >= threshold)]
                deepdanbooru_response = json.dumps(results),
                response = app.response_class(
                    response=deepdanbooru_response,
                    status=200,
                    mimetype='application/json'
                )
                return response

        return '''
        <!doctype html>
        <title>Upload new File</title>
        <h1>Upload new File</h1>
        <form method=post enctype=multipart/form-data>
          <p><input type=file name=file>
             <input type=submit value=Upload>
        </form>
        '''

    app.run(host=host, port=port)

if __name__ == '__main__':
    run_server(float(os.environ.get('THRESHOLD') or 0.5), '127.0.0.1', 12151, True if os.environ.get('CPU') else False, os.environ.get('AUTH_KEY'))
