# test_all_packages.py
print("Testing all installed packages...")

packages = [
    'pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn',
    'nltk', 'wordcloud', 'plotly', 'xgboost', 'lightgbm',
    'optuna', 'transformers', 'spacy'
]

for package in packages:
    try:
        __import__(package)
        print(f"✅ {package} - OK")
    except ImportError as e:
        print(f"❌ {package} - FAILED: {e}")

print("\n🎉 All packages tested! You're ready to start!")