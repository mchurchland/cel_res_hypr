import unittest
import numpy as np
import pandas as pd

from util.sensory_input import place_input_weights, sensory_annotation_map


class TestSensoryInput(unittest.TestCase):
    def setUp(self):
        self.dense=np.array([[1.],[-2.],[3.],[-4.]],dtype=np.float32)

    def test_unchanged_coefficients_preserve_order_sign_and_count(self):
        result=place_input_weights(self.dense,[0,2],[3,1],"unchanged_coefficients")
        np.testing.assert_array_equal(result,[[0.],[3.],[0.],[1.]])
        self.assertEqual(np.count_nonzero(result),2)
        np.testing.assert_array_equal(self.dense,[[1.],[-2.],[3.],[-4.]])

    def test_matched_norm_and_coefficients_across_placements(self):
        a=place_input_weights(self.dense,[1,3],[1,3],"matched_total_norm")
        b=place_input_weights(self.dense,[1,3],[0,2],"matched_total_norm")
        np.testing.assert_allclose(np.linalg.norm(a),np.linalg.norm(self.dense),rtol=1e-6)
        np.testing.assert_array_equal(a[[1,3]],b[[0,2]])
        self.assertEqual(float(a[1,0]/a[3,0]),.5)

    def test_full_placement_is_identity(self):
        for calibration in ("unchanged_coefficients","matched_total_norm"):
            np.testing.assert_array_equal(place_input_weights(self.dense,np.arange(4),np.arange(4),calibration),self.dense)

    def test_invalid_masks_and_calibrations(self):
        for source,target in [([],[]),([0,0],[1,2]),([0,1],[-1,2]),([0],[1,2]),([.5],[1]),([4],[1])]:
            with self.assertRaises(ValueError):
                place_input_weights(self.dense,source,target,"unchanged_coefficients")
        with self.assertRaises(ValueError):
            place_input_weights(self.dense,[0],[1],"unknown")
        with self.assertRaises(ValueError):
            place_input_weights(np.zeros((4,1)),[0],[1],"matched_total_norm")
        with self.assertRaises(ValueError):
            place_input_weights(np.array([[np.nan]]),[0],[0],"matched_total_norm")

    def test_annotation_policy_mapping_and_missing_names(self):
        metadata=pd.DataFrame([
            ["A","type","Sensory neuron","url"],
            ["B","type","Polymodal neuron (sensory neuron, motor neuron)","url"],
            ["C","type","Motor neuron, putative sensory neuron","url"],
            ["D","type","Interneuron","url"],
            ["MISSING","type","Sensory neuron","url"],
            ["D","Neurotransmitter","Sensory neuron","url"],
        ],columns=["Entity1","Relationship","Entity2","Evidence URL"])
        mapped,missing=sensory_annotation_map(metadata,["D","B","A","C"])
        self.assertEqual(mapped.neuron.tolist(),["B","A"])
        self.assertEqual(mapped.node_index.tolist(),[1,2])
        self.assertEqual(missing,["MISSING"])


if __name__=="__main__":
    unittest.main()
