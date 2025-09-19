import pyqtgraph as pg
from PyQt5.QtWidgets import QMainWindow, QAction, qApp, QApplication, QFileDialog, QWidget, QInputDialog, QHBoxLayout, QFrame, QSplitter


class myImageView(pg.ImageView):
    
    def __init__(self, parent=None, name="ImageView", view=None, imageItem=None, levelMode='mono', *args):
        super().__init__(parent, name, view, imageItem, levelMode, *args)
        self.link=None


    def linkSlider(self,anaImageView):
        self.link=anaImageView

    def timeLineChanged(self):
        if not self.ignorePlaying:
            self.play(0)

        (ind, time) = self.timeIndex(self.timeLine)
        if ind != self.currentIndex:
            self.currentIndex = ind
            self.updateImage()
            if self.link!=None:
                self.link.setCurrentIndex(ind)
        self.sigTimeChanged.emit(ind, time)



