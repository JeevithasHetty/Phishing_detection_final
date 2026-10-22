import unittest, tempfile, os
import numpy as np, pandas as pd
from helpers import *
from src.data.urls import split_url, hostname, registered_domain, normalise_url, is_ip_host
from src.data.psl import get_psl
from src.data import splits as S
from src.data.loading import validate_schema
from src.common import load_config

class TestUrls(unittest.TestCase):
    def test_space_after_scheme_and_ports(self):
        self.assertEqual(hostname("http:// 69.162.100.198"), "69.162.100.198")
        self.assertEqual(hostname("https://User:pw@Foo.Example.COM:8080/a?b=1"), "foo.example.com")
        self.assertEqual(hostname("http://"), "")
    def test_registered_domain_icann_vs_private(self):
        self.assertEqual(registered_domain("http://a.b.example.co.uk/x"), "example.co.uk")
        self.assertEqual(registered_domain("http://a.b.repl.co"), "repl.co")                 # ICANN-only: provider grouped
        self.assertEqual(registered_domain("http://a.b.repl.co", include_private=True), "b.repl.co")
        self.assertEqual(registered_domain("http://1.2.3.4/x"), "1.2.3.4")
    def test_psl_wildcard_and_exception(self):
        p = get_psl(); self.assertEqual(p.registered_domain("foo.bar.ck"), "foo.bar.ck")  # *.ck wildcard
        self.assertEqual(p.registered_domain("www.ck"), "www.ck")                          # !www.ck exception
    def test_normalise_conflict_key(self):
        self.assertEqual(normalise_url("HTTP://WWW.Foo.com/Bar/"), normalise_url("http://foo.com/bar"))
    def test_ip(self): self.assertTrue(is_ip_host("10.0.0.1")); self.assertFalse(is_ip_host("a.com"))

class TestSchemaAndLabels(unittest.TestCase):
    def test_valid(self):
        validate_schema(synth_urls(50), "x", load_config())
    def test_bad_label_rejected(self):
        d = synth_urls(50); d.loc[0, "label"] = 2
        with self.assertRaises(ValueError): validate_schema(d, "x", load_config())
    def test_missing_col(self):
        with self.assertRaises(ValueError): validate_schema(synth_urls(50).drop(columns=["domain"]), "x", load_config())
    def test_empty_url_rejected(self):
        d = synth_urls(50); d.loc[0, "url"] = ""
        with self.assertRaises(ValueError): validate_schema(d, "x", load_config())
    def test_label_mapping_1_is_phishing(self):
        d = synth_urls(300); self.assertTrue(d[d.label == 1].url.str.contains("secure-login").all())

class TestSplits(unittest.TestCase):
    def setUp(self):
        d = synth_urls(3000, 1); extra = d.sample(40, random_state=0).copy(); extra["label"] = 1 - extra["label"]
        self.raw = pd.concat([d, extra, d.iloc[:30]], ignore_index=True)   # conflicts + exact duplicates
        self.pool, self.ex = S.build_pool(self.raw)
    def test_pool_exclusions(self):
        self.assertGreater(self.ex["label_conflict_rows"], 0); self.assertGreater(self.ex["normalised_duplicates_dropped"], 0)
        self.assertFalse(self.pool["norm"].duplicated().any())
        self.assertTrue((self.pool.groupby("norm").label.nunique() == 1).all())
    def test_regdomain_disjoint(self):
        r = np.random.default_rng(0); pool = self.pool.copy()
        pool["rd"] = [f"d{i % 400}.com" for i in range(len(pool))]; pool["host"] = [f"h{i % 900}.d{i % 400}.com" for i in range(len(pool))]
        sp = S.make_split(pool, "regdomain", 5); v = S.verify_split(pool, sp, "regdomain")
        self.assertTrue(v["disjoint_at_level"]); self.assertEqual(set(sp), set(S.PARTS))
        sp2 = S.make_split(pool, "hostname", 5); self.assertTrue(S.verify_split(pool, sp2, "hostname")["disjoint_at_level"])
        self.assertFalse(S.verify_split(pool, sp2, "regdomain")["disjoint_at_level"])   # hostname split leaks registered domains
    def test_deterministic_and_checksum(self):
        a = S.make_split(self.pool, "regdomain", 3); b = S.make_split(self.pool, "regdomain", 3); c = S.make_split(self.pool, "regdomain", 4)
        self.assertTrue((a == b).all()); self.assertNotEqual(S.split_checksum(self.pool.row_id.values, a), S.split_checksum(self.pool.row_id.values, c))
    def test_random_stratified(self):
        sp = S.make_split(self.pool, "random", 1); rates = [self.pool.label[sp == p].mean() for p in S.PARTS]
        self.assertLess(max(rates) - min(rates), 0.02)
    def test_source_heldout_disjoint(self):
        pool = self.pool.copy(); pool["sources"] = np.where(np.arange(len(pool)) % 2 == 0, "A", "B")
        pool["rd"] = [f"d{i % 300}.com" for i in range(len(pool))]
        sp = S.source_heldout(pool, "A", "B", 1); v = S.verify_split(pool, sp, "regdomain")
        self.assertTrue(v["disjoint_at_level"]); self.assertTrue((pool.sources[sp == "test"] == "B").all()); self.assertTrue((pool.sources[sp == "train"] == "A").all())
if __name__ == "__main__": unittest.main()
