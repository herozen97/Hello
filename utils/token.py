class TOKEN():
    def __init__(self, MAX_LID):
        self.PAD = 0
        self.EOS = MAX_LID + 1   # end of seq
        self.SOS = MAX_LID + 2   # start of seq