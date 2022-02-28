import xml.dom.minidom

# get an empty doc
doc = xml.dom.minidom.Document()
# set a root node: annotation

# Fisrt Layer
root = doc.createElement('annotation')
doc.appendChild(root)
# End First

## Second Layer
nodeFolder = doc.createElement('folder')
nodeFileName = doc.createElement('filename')

nodeFolder.appendChild(doc.createTextNode('VOC2012'))
nodeFileName.appendChild(doc.createTextNode('2007_000027.jpg'))
root.appendChild(nodeFolder)
root.appendChild(nodeFileName)

### Third Layer
nodeSource = doc.createElement('source')
nodeDatabase = doc.createElement('database')
nodeDatabase.appendChild(doc.createTextNode('The VOC2007 Database'))

nodeAnnotation = doc.createElement('annotation')
nodeAnnotation.appendChild(doc.createTextNode('PASCAL VOC2007'))

nodeImage = doc.createElement('image')
nodeImage.appendChild(doc.createTextNode('flickr'))

nodeSource.appendChild(nodeDatabase)
nodeSource.appendChild(nodeAnnotation)
nodeSource.appendChild(nodeImage)
### End Third Layer

root.appendChild(nodeSource)
## End Second Layer

## Second Layer
nodeSize = doc.createElement('size')
nodeWidth = doc.createElement('width')
nodeHeight = doc.createElement('height')
nodeDepth = doc.createElement('depth')

### Third Layer
nodeWidth.appendChild(doc.createTextNode('231'))
nodeHeight.appendChild(doc.createTextNode('321'))
nodeDepth.appendChild(doc.createTextNode('3'))

nodeSize.appendChild(nodeWidth)
nodeSize.appendChild(nodeHeight)
nodeSize.appendChild(nodeDepth)
### End Third Layer
root.appendChild(nodeSize)
## End Second Layer

## Second Layer
nodeSegmentedLayer = doc.createElement('segmented')
nodeSegmentedLayer.appendChild(doc.createTextNode('0'))
root.appendChild(nodeSegmentedLayer)
## End Second Layer
Birds = {'bird1':[1,2,3,4],'bird2':[2,3,4,1],'bird3':[1,3,2,4]}

for i in Birds:
    nodeObject = doc.createElement('object')
    nodeName = doc.createElement('name')
    nodePose = doc.createElement('pose')
    nodeTruncated = doc.createElement('truncated')
    nodeDifficult = doc.createElement('difficult')
    nodeBndbox = doc.createElement('bndbox')

    nodeName.appendChild(doc.createTextNode(str(i)))
    nodePose.appendChild(doc.createTextNode('0'))
    nodeTruncated.appendChild(doc.createTextNode('1'))
    nodeDifficult.appendChild(doc.createTextNode('2'))

    nodeX_min = doc.createElement('xmin')
    nodeY_min = doc.createElement('ymin')
    nodeX_max = doc.createElement('xmax')
    nodeY_max = doc.createElement('ymax')

    nodeX_min.appendChild(doc.createTextNode(str(Birds[i][0])))
    nodeY_min.appendChild(doc.createTextNode(str(Birds[i][1])))
    nodeX_max.appendChild(doc.createTextNode(str(Birds[i][2])))
    nodeY_max.appendChild(doc.createTextNode(str(Birds[i][3])))

    nodeBndbox.appendChild(nodeX_min)
    nodeBndbox.appendChild(nodeY_min)
    nodeBndbox.appendChild(nodeX_max)
    nodeBndbox.appendChild(nodeY_max)

    nodeObject.appendChild(nodeName)
    nodeObject.appendChild(nodePose)
    nodeObject.appendChild(nodeTruncated)
    nodeObject.appendChild(nodeDifficult)
    nodeObject.appendChild(nodeBndbox)
    root.appendChild(nodeObject)





# write xml
fp = open('/Users/maojietang/Documents/Audubon_F21/Flex_Faster_RCNN/annotation.xml', 'w')
doc.writexml(fp, addindent ='\t', newl = '\n', encoding = 'utf-8')