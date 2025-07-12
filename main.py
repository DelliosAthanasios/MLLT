from translator import CodeTranslator
from gui import TrainingGUI
import warnings
warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage*")

if __name__ == "__main__":
    translator = CodeTranslator()
    gui = TrainingGUI(translator)
    gui.run() 