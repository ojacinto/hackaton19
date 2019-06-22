from image_process import open_image, resize_image, array_image, process_image_file
import pickle


class mlmodel(object):

    def __init__(self):
        self.classifier = self.load_pickle('best_classifier')
        self.team = 'Print Team'
        self.name = 'EnsmbleModel(ExtraTree - RandomForest)'
        #self.img_path = img_path
        #print(self.predict(self.img_path))

    def load_pickle(self, pickle_name):
        pickle_in = open(pickle_name, "rb")
        loaded = pickle.load(pickle_in)
        pickle_in.close()
        return loaded

    def get_image(self, imgpath):
        return process_image_file(imgpath)

    def predict(self, imgpath):
        image_to_predict = self.get_image(imgpath)
        probas = self.classifier.predict_proba(image_to_predict.reshape(1, -1)).tolist()[0]
        certainty = max(probas)
        pred = self.classifier.predict(image_to_predict.reshape(1, -1))
        predict = pred[0]
        if certainty < 0.4:
            predict = 'Undefined'
        top, rigth, bottom, left = (0, 0, 0, 0)
        probas = "Apple: {0}, Banana: {1}, Orange: {2}".format(probas[0], probas[1], probas[2])
        coords = top, rigth, bottom, left
        label = predict
        detected_fruits = [
            (label, certainty, coords, probas)
        ]
        return detected_fruits

#mlmodel('/home/sergio/work/src/hackaton_machine_learning19/validation/fresa.jpg')
