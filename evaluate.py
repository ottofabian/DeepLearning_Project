import data.cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as eval
import os

from data.cityscapesscripts.helpers.csHelpers import printError

args = eval.args

# image filter
args.groundTruthSearch = os.path.join(args.cityscapesPath, "gtFine", "val", "*", "*_gtFine_labelIds.png")

groundTruthImgList = eval.glob.glob(args.groundTruthSearch)

# only first 5 images of val set (example)
groundTruthImgList = groundTruthImgList

predictionImgList = []

if not groundTruthImgList:
    printError("Cannot find any ground truth images to use for evaluation. Searched for: {}".format(
        args.groundTruthSearch))
# get the corresponding prediction for each ground truth imag
for gt in groundTruthImgList:
    predictionImgList.append(eval.getPrediction(args, gt))

    # evaluate
eval.evaluateImgLists(predictionImgList, groundTruthImgList, args)

quit()
