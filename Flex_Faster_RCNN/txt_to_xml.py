import xml.dom.minidom
import os


# print(os.getcwd())
# path = Extract_Txt.Extract('/Users/maojietang/Downloads')
# print('P', path)
# path.print_()
# # path.get_path()
# i = path.Get_Info()
# print('I', i[1])
# #
def txt_convert_to_xml(filename, Image_shape, name, position):
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
    nodeFileName.appendChild(doc.createTextNode(filename))
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

    for i in range(len(name)):
        nodeObject = doc.createElement('object')
        nodeName = doc.createElement('name')
        nodePose = doc.createElement('pose')
        nodeTruncated = doc.createElement('truncated')
        nodeDifficult = doc.createElement('difficult')
        nodeBndbox = doc.createElement('bndbox')

        nodeName.appendChild(doc.createTextNode(str(name[i])))
        nodePose.appendChild(doc.createTextNode('0'))
        nodeTruncated.appendChild(doc.createTextNode('1'))
        nodeDifficult.appendChild(doc.createTextNode('2'))

        nodeX_min = doc.createElement('xmin')
        nodeY_min = doc.createElement('ymin')
        nodeX_max = doc.createElement('xmax')
        nodeY_max = doc.createElement('ymax')

        nodeX_min.appendChild(doc.createTextNode(str(position[i][0])))
        nodeY_min.appendChild(doc.createTextNode(str(position[i][1])))
        nodeX_max.appendChild(doc.createTextNode(str(position[i][2])))
        nodeY_max.appendChild(doc.createTextNode(str(position[i][3])))

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
    arg_output_dir = './Annotation_xml'
    if not os.path.exists(arg_output_dir):
        os.makedirs(arg_output_dir)
    fp = open(os.path.join(arg_output_dir, filename), 'w')
    doc.writexml(fp, addindent='\t', newl='\n', encoding='utf-8')

txt_convert_to_xml('File1.xml',[224,224],['Bird1'],[[1,2,3,4]])
