"""
gui/app.py
Простой Tkinter-интерфейс поверх готового core/-pipeline.
GUI не дублирует логику распознавания — только вызывает существующие функции.
"""
import sys
import os
import tkinter as tk
from tkinter import messagebox, filedialog

# Позволяет запускать файл напрямую (python gui/app.py) из корня проекта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import db_manager
from core import registration, recognition


class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("400x320")

        db_manager.init_db()

        tk.Label(root, text="Face Recognition Attendance", font=("Arial", 14)).pack(pady=15)

        tk.Label(root, text="Имя нового пользователя:").pack()
        self.name_entry = tk.Entry(root, width=30)
        self.name_entry.pack(pady=5)

        tk.Button(root, text="Зарегистрировать (камера)", width=30,
                  command=self.handle_register).pack(pady=5)
        tk.Button(root, text="Распознавание (live камера)", width=30,
                  command=self.handle_recognize_camera).pack(pady=5)
        tk.Button(root, text="Распознать по фото...", width=30,
                  command=self.handle_recognize_file).pack(pady=5)
        tk.Button(root, text="Посещаемость сегодня", width=30,
                  command=self.handle_show_attendance).pack(pady=5)

        self.status_label = tk.Label(root, text="", fg="blue")
        self.status_label.pack(pady=10)

    def handle_register(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Ошибка", "Введите имя пользователя.")
            return
        self.status_label.config(text=f"Регистрация '{name}'... (окно камеры)")
        self.root.update()
        success = registration.register_user(name)
        self.status_label.config(text="Готово." if success else "Прервано.")

    def handle_recognize_camera(self):
        self.status_label.config(text="Распознавание... (окно камеры)")
        self.root.update()
        recognition.recognize_from_camera()
        self.status_label.config(text="Сессия завершена.")

    def handle_recognize_file(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if not path:
            return
        result = recognition.recognize_from_image(path)
        if result is None:
            messagebox.showinfo("Результат", "Лицо не найдено.")
        else:
            messagebox.showinfo("Результат", f'{result["name"]} (similarity={result["similarity"]:.3f})')

    def handle_show_attendance(self):
        rows = db_manager.get_today_attendance()
        text = "\n".join(f"{n} — {t}" for n, t in rows) if rows else "Сегодня никто не отмечен."
        messagebox.showinfo("Посещаемость сегодня", text)


if __name__ == "__main__":
    root = tk.Tk()
    AttendanceApp(root).mainloop()
