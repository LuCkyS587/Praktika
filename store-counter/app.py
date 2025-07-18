import os
import cv2
import sqlite3
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from ultralytics import YOLO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from openpyxl import Workbook

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['REPORT_FOLDER'] = 'static/reports'

# Создаем папки, если их нет
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPORT_FOLDER'], exist_ok=True)

# Загружаем модель YOLOv8 для детекции людей
model = YOLO('yolov8n.pt')

# Инициализация базы данных
def init_db():
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS requests
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 timestamp DATETIME,
                 filename TEXT,
                 count INTEGER,
                 result_path TEXT)''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Сохраняем оригинальный файл
    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Обработка изображения/видео
    if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        result = process_image(filepath, filename)
    elif file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        result = process_video(filepath, filename)
    else:
        return jsonify({'error': 'Unsupported file format'}), 400

    return jsonify(result)

def process_image(filepath, filename):
    # Чтение и обработка изображения
    img = cv2.imread(filepath)
    results = model(img)
    
    # Визуализация результатов
    output_img = results[0].plot()
    result_filename = f"result_{filename}"
    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    cv2.imwrite(result_path, output_img)
    
    # Подсчет людей (класс 0 в COCO)
    count = sum(1 for box in results[0].boxes if box.cls == 0)
    
    # Сохранение в БД
    save_to_db(filename, count, result_path)
    
    return {
        'count': count,
        'result_url': f'/static/results/{result_filename}',
        'filename': filename
    }

def process_video(filepath, filename):
    # Обработка первого кадра видео
    cap = cv2.VideoCapture(filepath)
    success, frame = cap.read()
    if not success:
        return {'error': 'Failed to read video'}
    
    # Сохраняем первый кадр
    frame_filename = f"frame_{filename.split('.')[0]}.jpg"
    frame_path = os.path.join(app.config['UPLOAD_FOLDER'], frame_filename)
    cv2.imwrite(frame_path, frame)
    
    # Обрабатываем кадр как изображение
    return process_image(frame_path, frame_filename)

def save_to_db(filename, count, result_path):
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute("INSERT INTO requests (timestamp, filename, count, result_path) VALUES (?, ?, ?, ?)",
              (datetime.now(), filename, count, result_path))
    conn.commit()
    conn.close()

@app.route('/report/<report_type>')
def generate_report(report_type):
    # Получаем данные из БД
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute("SELECT timestamp, filename, count FROM requests")
    data = c.fetchall()
    conn.close()
    
    # Генерируем отчет
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    
    if report_type == 'pdf':
        report_path = os.path.join(app.config['REPORT_FOLDER'], f'report_{timestamp}.pdf')
        generate_pdf(data, report_path)
    elif report_type == 'excel':
        report_path = os.path.join(app.config['REPORT_FOLDER'], f'report_{timestamp}.xlsx')
        generate_excel(data, report_path)
    else:
        return jsonify({'error': 'Invalid report type'}), 400
    
    return send_file(report_path, as_attachment=True)

def generate_pdf(data, path):
    c = canvas.Canvas(path, pagesize=letter)
    c.setFont("Helvetica", 12)
    
    # Заголовок
    c.drawString(100, 750, "Отчет по подсчету посетителей")
    c.drawString(100, 730, f"Сгенерирован: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.line(100, 725, 500, 725)
    
    # Данные
    y = 700
    for row in data:
        timestamp, filename, count = row
        c.drawString(100, y, f"{timestamp} | {filename} | Посетителей: {count}")
        y -= 20
        if y < 50:
            c.showPage()
            y = 750
    
    c.save()

def generate_excel(data, path):
    wb = Workbook()
    ws = wb.active
    ws.title = "Отчет"
    
    # Заголовки
    ws.append(["Дата и время", "Файл", "Количество посетителей"])
    
    # Данные
    for row in data:
        ws.append(row)
    
    wb.save(path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)