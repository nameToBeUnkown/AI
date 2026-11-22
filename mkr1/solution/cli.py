#!/usr/bin/env python3
"""CLI interface for TechPrep AI — Mock Interview & Question Predictor (Cloud Edition)."""

import argparse
import json
import os
import sys

if sys.stdout:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr:
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import TechPrepEngine, DEFAULT_BASE_URL, DEFAULT_API_KEY, DEFAULT_MODEL

def main():
    parser = argparse.ArgumentParser(description="TechPrep AI — генератор питань до співбесіди через Cloud API.")
    parser.add_argument("--vacancy-file", "-f", type=str, help="Шлях до текстового файлу з описом вакансії")
    parser.add_argument("--count", "-c", type=int, default=5, help="Кількість запитань для генерації (за замовчуванням 5)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--export-md", type=str, help="Шлях для збереження шпаргалки у форматі Markdown")
    parser.add_argument("--interactive", action="store_true", help="Запустити інтерактивний режим симуляції співбесіди")

    args = parser.parse_args()

    if not args.api_key:
        print("⚠️ Увага: Не вказано API ключ. Встановіть змінну оточення GROQ_API_KEY або OPENAI_API_KEY, або передайте --api-key.")

    vacancy_text = ""
    if args.vacancy_file:
        if not os.path.exists(args.vacancy_file):
            print(f"❌ Помилка: файл вакансії не знайдено: {args.vacancy_file}")
            sys.exit(1)
        with open(args.vacancy_file, "r", encoding="utf-8") as f:
            vacancy_text = f.read()
    else:
        print("Введіть або вставте текст вакансії (Ctrl+Z або Ctrl+D для завершення):")
        try:
            vacancy_text = sys.stdin.read()
        except Exception:
            pass

    if not vacancy_text.strip():
        print("❌ Помилка: текст вакансії порожній.")
        sys.exit(1)

    print(f"\n🚀 Підключення до Cloud API (Модель: {args.model}, Endpoint: {args.base_url})...")
    engine = TechPrepEngine(base_url=args.base_url, api_key=args.api_key, model=args.model)

    print("\n⏳ Крок 1: Хмарний аналіз вимог вакансії та вилучення стеку...")
    try:
        parsed_data, parse_usage = engine.parse_vacancy(vacancy_text)
        print(f"✅ Позиція: {parsed_data.get('role')} ({parsed_data.get('level')})")
        print(f"📦 Стек: {', '.join(parsed_data.get('tech_stack', []))}")
        print(f"🏢 Домен: {parsed_data.get('domain')}")
        print(f"⏱️ Час аналізу: {parse_usage.get('latency_sec')} с ({parse_usage.get('total_tokens')} токенів)")
    except Exception as e:
        print(f"❌ Помилка аналізу вакансії: {e}")
        sys.exit(1)

    print(f"\n⏳ Крок 2: Хмарна генерація {args.count} цільових практичних запитань...")
    try:
        questions, gen_usage = engine.generate_interview_questions(parsed_data, count=args.count)
        print(f"✅ Згенеровано {len(questions)} запитань за {gen_usage.get('latency_sec')} с.")
    except Exception as e:
        print(f"❌ Помилка генерації запитань: {e}")
        sys.exit(1)

    print("\n" + "="*70)
    print("📋 ПЕРЕЛІК ЦІЛЬОВИХ ЗАПИТАНЬ ТЕХНІЧНОГО ІНТЕРВ'Ю")
    print("="*70)
    for q in questions:
        print(f"\n[#{q.get('id')}] Категорія: {q.get('category')}")
        print(f"❓ {q.get('question')}")
        print(f"🎯 Мета ліда: {q.get('interviewer_goal')}")

    if args.export_md:
        md = engine.export_cheat_sheet_markdown(parsed_data, questions)
        with open(args.export_md, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"\n💾 Шпаргалку збережено до {args.export_md}")

    if args.interactive and questions:
        print("\n" + "="*70)
        print("🎙️ ІНТЕРАКТИВНИЙ СИМУЛЯТОР СПІВБЕСІДИ (MOCK INTERVIEW)")
        print("="*70)
        q = questions[0]
        print(f"\nЗапитання ліда: {q.get('question')}")
        print("Введіть вашу відповідь (або 'skip' для пропуску):")
        ans = input(">> ").strip()
        if ans and ans.lower() != "skip":
            print("\n⏳ Оцінювання відповіді в ролі Tech Lead...")
            eval_data, eval_usage = engine.evaluate_candidate_answer(q, ans)
            print(f"\n📊 Оцінка: {eval_data.get('score')}/10 | Вердикт: {eval_data.get('verdict')}")
            print("\n🟢 Сильні сторони (Green Flags):")
            for gf in eval_data.get("green_flags", []):
                print(f"  + {gf}")
            print("\n🔴 Ризики та неточності (Red Flags):")
            for rf in eval_data.get("red_flags", []):
                print(f"  - {rf}")
            print(f"\n💡 Ідеальна відповідь на інтерв'ю:\n{eval_data.get('golden_answer')}")

if __name__ == "__main__":
    main()
