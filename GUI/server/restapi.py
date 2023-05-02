from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import os
from flask_cors import CORS
from werkzeug.utils import secure_filename
import script

app = Flask(__name__)

upload_folder = os.path.join(os.path.dirname(__file__), 'upload/')

api = Api(app)
CORS(app)
@app.route("/")

class Images(Resource):
    
    '''
        Causes the image to be run through the model and returns the data that is returned by said model
        Inputs:
              newImg - the image that has to be run through the model
        Returns:
            The output from the classifier plus the number of birds found in the image
    '''
    def post(self):
        files = request.files
        image = files.get('newIMG')

        # Saves the image to a location in the working directory
        filename = secure_filename(image.filename)
        path = os.path.join(upload_folder, filename)
        print("Path saved: " + filename)
        image.save(path)

        # Sends the image to the classifier to be tested
        nameArray, dataArray = script.bird_classifier(path)

        return {'data': dataArray,
                'bird_names': nameArray,
                'num_birds': len(nameArray)}, 200

class Delete(Resource):

    '''
        Deletes all images generated for an image that is now done being used
        Inputs:
            birdNum - The number of images that need to be deleted
            fileName - The filename of the original image that needs to be deleted
    '''
    def post(self):
        data = request.json
        num = data['birdNum']
        name = data['fileName']

        os.remove(os.path.join(upload_folder, name))
        
        for i in range(1, num):
            string1 = 'bird' + str(i) + '.jpg'
            string2 = 'expanded_bird' + str(i) + '.jpg'
            os.remove(os.path.join(upload_folder, string1))
            os.remove(os.path.join(upload_folder, string2))


#api.add_resource(Annotations, '/annotations')
api.add_resource(Images, '/images')
api.add_resource(Delete, '/delete')

if __name__ == '__main__':
    app.run()  # run our Flask app