import unittest
import pickle
import os
import shutil
import nlp
import random

from sklearn.metrics import f1_score

import sys
sys.path.append('../')

from answer_engine import *

TESTSET_PATH = '../datasets'

class AnswerEngineTests(unittest.TestCase):

    # Test the question answering answer scoring system.
    # We test how well the it scores an the "most" correct answer 
    # using both ML but also hand coded rules
    def test_qa_cs_system(self):
        # Answering pipeline
        self.pipeline = pipeline(
            model_name='ktrapeznikov/albert-xlarge-v2-squad-v2',
            use_custom_pipeline=True,
            from_google_bucket=False)

        data = 'Denmark, officially the Kingdom of Denmark, is a Nordic country in Northwest Europe. Denmark proper, which is the southernmost of the Scandinavian countries, consists of a peninsula, Jutland, and an archipelago of 443 named islands, with the largest being Zealand, Funen and the North Jutlandic Island.'
        question = 'what is denmark'

        # Do question answering and article scoring
        pred = self.pipeline(
            contexts=[data], 
            question=question)

        pred_answer = pred[0][0]
        pred_score = pred[0][1]
        pred_crediability = pred[0][1]

        # Test if the output has the right output format
        # Tests if the dtype is correct
        self.assertEqual(type(pred_answer), str)
        self.assertEqual(type(pred_score), float)
        self.assertEqual(type(pred_crediability), int)

        # Tests if the score has the rigth length
        self.assertEqual(round(pred_score,1), pred_score)

    # Test file upload, download and delete functions
    def test_google_storage(self):
        # Create test files
        if not os.path.exists('test_files'):
            os.makedirs('test_files')
        with open(
            os.path.join('test_files', 'file.txt'), 'w') as f:
            f.write('Hejsa')

        bucket = google_storage.connect_google_cloud_bucket(
            bucket_name='sorcely-models', 
            base_path='../answer_engine/')

        # Checks we get the prefered output
        self.assertEqual(
            str(type(bucket)), '<class \'google.cloud.storage.bucket.Bucket\'>')

        # We can't test if the file was actually uploaded to the cloud
        # However we can try to download it again
        google_storage.upload_file_to_google_storage(
            bucket=bucket, 
            storage_name='sorcely-models/test/file.txt', 
            file_name='test_files/file.txt')

        # Downloads the file
        google_storage.download_file_from_google_storage(
            bucket=bucket, 
            storage_name='sorcely-models/test/file.txt', 
            file_name='test_files/file.txt')

        # Delete file from google cloud storage bucket
        google_storage.delete_file_from_google_storage(
            bucket=bucket, 
            storage_name='sorcely-models/test/file.txt')

if __name__ == '__main__':
    unittest.main()
