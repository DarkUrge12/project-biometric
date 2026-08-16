"""
main.py
CLI-точка входа: меню регистрации, распознавания, посещаемости.
"""
from database import db_manager
from core import registration, recognition


def print_menu():
    print("\n--- Face Recognition Attendance System ---")
    print("1. Зарегистрировать пользователя")
    print("2. Распознавание (камера, live)")
    print("3. Распознать по фото из файла")
    print("4. Посещаемость за сегодня")
    print("5. Выход")


def main():
    db_manager.init_db()

    while True:
        print_menu()
        choice = input("Выберите пункт: ").strip()

        if choice == "1":
            name = input("Имя пользователя: ").strip()
            if name:
                registration.register_user(name)
            else:
                print("Имя не может быть пустым.")

        elif choice == "2":
            recognition.recognize_from_camera()

        elif choice == "3":
            path = input("Путь к изображению: ").strip()
            try:
                result = recognition.recognize_from_image(path)
            except FileNotFoundError as e:
                print(e)
                continue
            if result is None:
                print("Лицо не найдено.")
            else:
                print(f'Результат: {result["name"]} (similarity={result["similarity"]:.3f})')

        elif choice == "4":
            rows = db_manager.get_today_attendance()
            if not rows:
                print("Сегодня никто не отмечен.")
            else:
                for name, time in rows:
                    print(f"  {name} — {time}")

        elif choice == "5":
            break

        else:
            print("Неверный пункт меню.")


if __name__ == "__main__":
    main()
