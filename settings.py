import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
MODELS_ROOT = os.path.join(PROJECT_ROOT, 'models')
TB_LOGS_ROOT = os.path.join(PROJECT_ROOT, 'tb_logs')

if not os.path.exists(MODELS_ROOT):
    os.mkdir(MODELS_ROOT)
if not os.path.exists(TB_LOGS_ROOT):
    os.mkdir(TB_LOGS_ROOT)
