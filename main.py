from translator import CodeTranslator
from gui import TrainingGUI

if __name__ == "__main__":
    translator = CodeTranslator()
    gui = TrainingGUI(translator)
    gui.run() 