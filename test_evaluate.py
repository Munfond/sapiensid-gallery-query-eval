"""Tests for dataset labels and ranking metrics; no model dependencies needed."""
from pathlib import Path
import tempfile
import unittest

from evaluate import scan_dataset, score_one, summarize


class EvaluationTests(unittest.TestCase):
    def test_labels_keep_leading_zeros_and_ignore_query_filename(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'gallery').mkdir()
            (root / 'gallery/001.jpg').write_bytes(b'gallery')
            (root / 'query/001/session1').mkdir(parents=True)
            (root / 'query/001/session1/2_0.jpg').write_bytes(b'query')
            gallery, queries = scan_dataset(root)
            self.assertEqual(gallery[0]['person_id'], '001')
            self.assertEqual(queries[0]['person_id'], '001')

    def test_missing_id_stops_closed_set_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'gallery').mkdir()
            (root / 'gallery/001.jpg').write_bytes(b'gallery')
            (root / 'query/002').mkdir(parents=True)
            (root / 'query/002/a.jpg').write_bytes(b'query')
            with self.assertRaisesRegex(ValueError, 'missing from gallery'):
                scan_dataset(root)

    def test_duplicate_gallery_id_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'gallery').mkdir()
            (root / 'query').mkdir()
            (root / 'gallery/001.jpg').write_bytes(b'one')
            (root / 'gallery/001.png').write_bytes(b'two')
            with self.assertRaisesRegex(ValueError, 'Duplicate gallery IDs'):
                scan_dataset(root)

    def test_known_rank_and_failure_denominator(self):
        gallery = [dict(path=f'gallery/{identity}.jpg', person_id=identity) for identity in ['A', 'B', 'C']]
        query = dict(path='query/A/1.jpg', person_id='A')
        rank2 = score_one(query, gallery, [0.8, 0.9, 0.1])
        self.assertEqual(rank2['true_rank'], 2)
        self.assertFalse(rank2['correct'])
        self.assertEqual(rank2['reciprocal_rank'], 0.5)
        rank1 = score_one(query, gallery, [0.9, 0.8, 0.1])
        failed = score_one(dict(path='query/B/bad.jpg', person_id='B'), gallery, error='decode failed')
        metrics, people = summarize([rank2, rank1, failed], len(gallery))
        self.assertAlmostEqual(metrics['top1_accuracy'], 1/3)
        self.assertAlmostEqual(metrics['top5_accuracy'], 2/3)
        self.assertAlmostEqual(metrics['mAP'], 0.5)
        self.assertAlmostEqual(metrics['macro_top1_accuracy'], 0.25)
        self.assertEqual(metrics['failed_queries'], 1)
        self.assertEqual(metrics['effective_top5_k'], 3)

    def test_rank_six_is_not_top5(self):
        gallery = [dict(path=f'gallery/{i}.jpg', person_id=str(i)) for i in range(6)]
        result = score_one(dict(path='q.jpg', person_id='0'), gallery, [0, 1, 2, 3, 4, 5])
        self.assertEqual(result['true_rank'], 6)
        self.assertFalse(result['top5_correct'])

    def test_tie_break_does_not_use_ground_truth(self):
        gallery = [dict(path='gallery/A.jpg', person_id='A'), dict(path='gallery/B.jpg', person_id='B')]
        result = score_one(dict(path='q.jpg', person_id='B'), gallery, [0.5, 0.5])
        self.assertEqual(result['predicted_id'], 'A')

    def test_invalid_scores_rejected(self):
        with self.assertRaises(ValueError):
            score_one(dict(path='q.jpg', person_id='A'), [dict(path='g.jpg', person_id='A')], [float('nan')])


if __name__ == '__main__':
    unittest.main()
