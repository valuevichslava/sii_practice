from flask import Flask, render_template, request, send_file
import os
import cv2
import json
from datetime import datetime
import pandas as pd
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import subprocess

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['REPORTS_FOLDER'] = 'reports'
app.config['HISTORY_FILE'] = 'history.json'

model = YOLO('yolov8n.pt')  # Загрузка YOLOv8 с трекингом

def update_history(filename, file_type, unique_cows):
    entry = {
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'filename': filename,
        'file_type': file_type,
        'unique_cows': unique_cows
    }
    
    try:
        with open(app.config['HISTORY_FILE'], 'r+') as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
            history.append(entry)
            f.seek(0)
            json.dump(history, f, indent=4)
    except FileNotFoundError:
        with open(app.config['HISTORY_FILE'], 'w') as f:
            json.dump([entry], f, indent=4)

def process_image(img_path):
    try:
        # Предсказание без трекинга
        results = model.predict(
            source=img_path,
            classes=19,
            conf=0.5,
            imgsz=640
        )
        
        # Сохранение результата
        output_path = os.path.join(app.config['RESULT_FOLDER'], os.path.basename(img_path))
        results[0].save(filename=output_path)
        
        # Подсчет коров
        cow_count = len(results[0].boxes)
        
        return cow_count, output_path
        
    except Exception as e:
        print(f"Image processing error: {str(e)}")
        return 0, None

def process_video(video_path):
    # Оригинальный путь для обработки
    orig_output_path = os.path.join(app.config['RESULT_FOLDER'], os.path.basename(video_path))
    orig_output_path = orig_output_path.replace('\\', '/')  # Для Windows
    
    # Путь для совместимого видео
    compatible_path = os.path.join(app.config['RESULT_FOLDER'], 
                                f"compat_{os.path.basename(video_path)}")
    compatible_path = compatible_path.replace('\\', '/')
    
    # Веб-пути для доступа из HTML
    web_orig_path = orig_output_path.replace('static/', '/static/')
    web_compat_path = compatible_path.replace('static/', '/static/')

    # Обработка видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    # Получаем параметры видео
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Инициализируем VideoWriter
    writer = cv2.VideoWriter(
        orig_output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height)
    )

    unique_ids = set()
    tracker_config = os.path.join('ultralytics', 'cfg', 'trackers', 'botsort.yaml')

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Трекинг объектов
            results = model.track(
                frame,
                persist=True,
                classes=19,
                tracker=tracker_config
            )

            # Сбор уникальных ID
            if results[0].boxes.id is not None:
                current_ids = results[0].boxes.id.cpu().numpy().astype(int)
                unique_ids.update(current_ids)

            # Визуализация и запись
            annotated_frame = results[0].plot()
            writer.write(annotated_frame)
    finally:
        cap.release()
        writer.release()

    # Конвертация в совместимый формат
    try:
        subprocess.run(
            [
                'ffmpeg',
                '-y',
                '-i', orig_output_path,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-movflags', '+faststart',
                '-crf', '23',
                compatible_path
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr}")
        return len(unique_ids), web_orig_path

    return len(unique_ids), web_compat_path

@app.route('/', methods=['GET', 'POST'])
def index():

    history = []
    # Загружаем историю при любой загрузке страницы
    if os.path.exists(app.config['HISTORY_FILE']):
        with open(app.config['HISTORY_FILE'], 'r') as f:
            try:
                history = json.load(f)
                # Сортируем по дате в обратном порядке
                history = sorted(history, key=lambda x: x['date'], reverse=True)
            except json.JSONDecodeError:
                pass

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
            
            file_type = 'video' if filename.lower().endswith(('.mp4', '.avi', '.mov')) else 'photo'
            
            if file_type == 'photo':
                count, result_path = process_image(upload_path)
            else:
                count, result_path = process_video(upload_path)
            
            update_history(filename, file_type, count)
            
            return render_template('index.html', 
                             result_file=result_path,
                             file_type=file_type,
                             detection_count=count,
                             history=history)  # Добавляем историю в вывод
    
    return render_template('index.html', history=history)

@app.route('/report')
def generate_report():
    if not os.path.exists(app.config['HISTORY_FILE']):
        return "No history data"
    
    with open(app.config['HISTORY_FILE'], 'r') as f:
        history = json.load(f)
    
    df = pd.DataFrame(history)
    df = pd.DataFrame(history)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date', ascending=False)
    excel_path = os.path.join(app.config['REPORTS_FOLDER'], 'cow_report.xlsx')
    
    # Добавлен параметр engine
    df.to_excel(excel_path, index=False, engine='openpyxl')
    
    return send_file(excel_path, as_attachment=True)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
    os.makedirs(app.config['REPORTS_FOLDER'], exist_ok=True)
    app.run(debug=True)