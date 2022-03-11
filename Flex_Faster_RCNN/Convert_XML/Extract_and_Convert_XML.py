import pandas as pd
import numpy as np
import os
import xml.dom.minidom


def txt_convert_to_xml(filename, name, position):
    """
        Convert txt to xml and save，
        Args:
            filename: txt_file's path
            name: txt_file's name
            position: each bird's position Inf
        Returns:

    """
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

    nodeFolder.appendChild(doc.createTextNode('Audubon2022'))
    nodeFileName.appendChild(doc.createTextNode(filename))
    root.appendChild(nodeFolder)
    root.appendChild(nodeFileName)

    ### Third Layer
    nodeSource = doc.createElement('source')
    nodeDatabase = doc.createElement('database')
    nodeDatabase.appendChild(doc.createTextNode('The Audubon2022 Database'))

    nodeAnnotation = doc.createElement('annotation')
    nodeAnnotation.appendChild(doc.createTextNode('Audubon2022 Spring'))

    nodeImage = doc.createElement('image')
    nodeImage.appendChild(doc.createTextNode('None'))

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
    path = '/Users/maojietang/Documents/Audubon_F21/Flex_Faster_RCNN'
    filename = filename.split('.')[0] + '.xml'
    filename = filename.split('/')[-1]
    arg_output_dir = os.path.join(path, 'Annotation_xml')
    print(os.path.join(arg_output_dir, filename))
    if not os.path.exists(arg_output_dir):
        os.makedirs(arg_output_dir)
    fp = open(os.path.join(arg_output_dir, filename), 'w')
    doc.writexml(fp, addindent='\t', newl='\n', encoding='utf-8')


class Extract(object):
    def __init__(self, txt_path):
        # txt_path = '/Users/maojietang/Downloads'
        self.File_path = None
        self.xml_list = None
        self.txt_path = txt_path

    def get_path(self):
        if os.path.exists(os.path.join(self.txt_path, "Annotations")) is False:
            raise FileNotFoundError("Annotation dose not in path:'{}'.".format(self.txt_path))
        self.File_path = self.txt_path + '/Annotation'

        self.xml_list = [os.path.join(self.File_path, line.strip())
                         for line in os.listdir(self.File_path) if (line.strip()[-3:]) == 'bbx']

    def get_info(self):
        self.get_path()
        for i in self.xml_list:
            file = pd.read_csv(i)
            # get AI Class
            name = file.iloc[:, 0].values

            # get Species
            species = file.iloc[:, 1].values

            # get position info
            position = file.iloc[:, 2:].values

            # Need X_min, Y_min, X_max, Y_max
            X_min = position[:, 0:1]
            Y_min = position[:, 1:2]
            X_max = position[:, 0:1] + position[:, 2:3]
            Y_max = position[:, 1:2] + position[:, 3:4]
            Position = np.hstack((X_min, Y_min, X_max, Y_max))
            filename = i.split('.')[0]
            filename = filename.split('/')[-1] + '.jpg'

            txt_convert_to_xml(filename, name, Position)
if __name__ == '__main__':
    A = Extract('/Users/maojietang/Downloads/Test')
    A.get_info()
