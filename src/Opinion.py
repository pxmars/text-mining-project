"""
Class modeling the Opinion object
The Opinion attributes are the opinion holder, the post date, the tweet text,
the target entity, teh target aspect and the sentiment orientation
"""


class Opinion(object):
    def __str__(self) -> str:
        return "opinionHolder: " + self.opinionHolder + ", postDate: " + self.postDate + ", targetEntity: " + \
               self.targetEntity + ", targetAspect: " + self.targetAspect + ", SO: " + self.SO.__str__()

    def __init__(self, opinionHolder, postDate, text, targetEntity="", targetAspect="", SO=0):
        """
        Opinion constructor
        :param opinionHolder:
        :param postDate:
        :param text:
        :param targetEntity:
        :param targetAspect:
        :param SO:
        """
        self.opinionHolder = opinionHolder
        self.postDate = postDate
        self.targetEntity = targetEntity
        self.targetAspect = targetAspect
        self.SO = SO
        self.text = text

    def setTargetEntity(self, targetEntity):
        self.targetEntity = targetEntity

    def setTargetAspect(self, targetAspect):
        self.targetAspect = targetAspect

    def setSO(self, SO):
        self.SO = SO

    def getTargetEntity(self):
        return self.targetEntity

    def getTargetAspect(self):
        return self.targetAspect

    def getSO(self):
        return self.SO

    def getText(self):
        return self.text
