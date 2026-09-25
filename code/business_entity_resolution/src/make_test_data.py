#!/usr/bin/env python3
"""
Generate a small synthetic dataset to smoke-test the pipeline.

Creates dataset/train/ and dataset/test/ with realistic-looking records
for US and India (train) and France (test), including noise patterns
the pipeline must handle: abbreviations, typos, missing fields, multiple
matches per S1, and singletons.

Writes to <repo>/dataset_fake/ (NOT dataset/, which holds the real data)
and refuses to overwrite existing files. Run from the repo root:
    python code/business_entity_resolution/src/make_test_data.py
    python code/business_entity_resolution/src/run_pipeline.py --mode validate \
        --data-dir dataset_fake --model-dir dataset_fake/models --output-dir dataset_fake/output
"""

import argparse
import os
import sys

# Use the same path trick as run_pipeline so config finds the repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def write_tsv(path, header, rows):
    """Write a TSV file from a list of row tuples (never overwrites a file)."""
    if os.path.exists(path):
        sys.exit(f"  ✗ {path} already exists — refusing to overwrite (real data?).")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(str(x) for x in row) + "\n")
    print(f"  ✓ {path} ({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser(description="Generate a tiny synthetic dataset")
    parser.add_argument("--out-dir", default=os.path.join(config.REPO_ROOT, "dataset_fake"),
                        help="Where to write train/ and test/ (default: <repo>/dataset_fake)")
    args = parser.parse_args()
    config.set_paths(args.out_dir)

    print("Generating synthetic test dataset...\n")

    # ── Train Source 1 (S1): 20 records ──
    train_s1 = [
        ("S1-00001", "Acme Corporation", "123 Main St, Springfield, IL 62701", "US"),
        ("S1-00002", "Tata Consultancy Services Pvt Ltd", "Plot No 56, Rajiv Gandhi Infotech Park, Hinjewadi, Pune 411057", "India"),
        ("S1-00003", "Global Tech Solutions Inc", "456 Innovation Blvd, Austin, TX 78701", "US"),
        ("S1-00004", "Infosys Limited", "44 Electronics City, Hosur Road, Bangalore 560100", "India"),
        ("S1-00005", "American Standard Manufacturing Co", "789 Industrial Pkwy, Detroit, MI 48201", "US"),
        ("S1-00006", "Reliance Industries Ltd", "Maker Chambers IV, Nariman Point, Mumbai 400021", "India"),
        ("S1-00007", "Johnson & Associates", "321 Oak Avenue, Portland, OR 97201", "US"),
        ("S1-00008", "Wipro Technologies Private Limited", "Doddakannelli, Sarjapur Road, Bangalore 560035", "India"),
        ("S1-00009", "Pacific Trading Company", "555 Harbor Drive, San Diego, CA 92101", "US"),
        ("S1-00010", "Mahindra & Mahindra Ltd", "Gateway Building, Apollo Bunder, Mumbai 400001", "India"),
        ("S1-00011", "Summit Healthcare Systems", "100 Medical Center Drive, Boston, MA 02215", "US"),
        ("S1-00012", "Bharti Airtel Limited", "Airtel Center, Plot No 16, Udyog Vihar, Gurgaon 122016", "India"),
        ("S1-00013", "Valley Construction LLC", "200 Builder Rd, Sacramento, CA 95814", "US"),
        ("S1-00014", "Sun Pharmaceutical Industries Ltd", "SPARC, Tandalja, Baroda 390020", "India"),
        ("S1-00015", "Northwest Insurance Group", "300 Tower Blvd, Seattle, WA 98101", "US"),
        ("S1-00016", "HCL Technologies Ltd", "Plot 3A, Sector 126, Noida 201304", "India"),
        ("S1-00017", "Golden State Logistics Inc", "400 Freight Way, Los Angeles, CA 90001", "US"),
        ("S1-00018", "Larsen & Toubro Limited", "LnT House, Ballard Estate, Mumbai 400001", "India"),
        ("S1-00019", "Premier Auto Parts", "500 Motor Ave, Chicago, IL 60601", "US"),
        ("S1-00020", "HDFC Bank Ltd", "HDFC Bank House, Senapati Bapat Marg, Lower Parel, Mumbai 400013", "India"),
    ]

    # ── Train Source 2 (S2): noisy variants of some S1 records + unmatched ──
    train_s2 = [
        # Matches to S1
        ("S2-00001", "ACME Corp", "123 Main Street, Springfield IL 62701", "US"),  # S1-00001
        ("S2-00002", "Tata Consultancy Svcs Pvt. Ltd.", "Plot 56, Rajiv Gandhi IT Park, Hinjewadi, Pune, 411057", "India"),  # S1-00002
        ("S2-00003", "Global Tech Solns", "456 Innovation Boulevard, Austin TX 78701", "US"),  # S1-00003
        ("S2-00004", "Infosys Ltd", "44 Electronic City, Hosur Rd, Bangalore 560100", "India"),  # S1-00004
        ("S2-00005", "American Std Mfg Company", "789 Industrial Parkway, Detroit, MI 48201", "US"),  # S1-00005
        ("S2-00006", "Reliance Inds Limited", "Maker Chambers, Nariman Point, Mumbai, 400021", "India"),  # S1-00006
        ("S2-00007", "Johnson and Associates", "321 Oak Ave, Portland OR 97201", "US"),  # S1-00007
        ("S2-00008", "Wipro Tech Pvt Ltd", "Doddakannelli, Sarjapur Rd, Bangalore 560035", "India"),  # S1-00008
        # No match to any S1
        ("S2-00009", "Random Corp LLC", "999 Nowhere Lane, Smalltown, KS 67401", "US"),
        ("S2-00010", "Bharat Steels Private Limited", "Industrial Area Phase 2, Ludhiana 141003", "India"),
        ("S2-00011", "Sunshine Foods Inc", "777 Harvest Rd, Fresno CA 93701", "US"),
    ]

    # ── Train Source 3 (S3): another set of noisy variants ──
    train_s3 = [
        # Matches to S1
        ("S3-00001", "Acme Corp.", "123 Main St., Springfield, IL", "US"),  # S1-00001 (second match)
        ("S3-00002", "TCS Private Limited", "Rajiv Gandhi Infotech Park, Hinjewadi Phase 1, Pune 411057", "India"),  # S1-00002
        ("S3-00003", "Pacific Trading Co", "555 Harbor Dr, San Diego 92101", "US"),  # S1-00009
        ("S3-00004", "Mahindra and Mahindra Limited", "Gateway Bldg, Apollo Bunder, Mumbai, 400001", "India"),  # S1-00010
        ("S3-00005", "Summit Healthcare Sys", "100 Medical Ctr Dr, Boston MA 02215", "US"),  # S1-00011
        ("S3-00006", "Bharti Airtel Ltd", "Plot 16, Udyog Vihar Phase IV, Gurgaon, 122016", "India"),  # S1-00012
        ("S3-00007", "HCL Tech Limited", "Plot 3A, Sector 126, Noida, 201304", "India"),  # S1-00016
        ("S3-00008", "L&T Ltd", "LT House, Ballard Estate, Mumbai, 400001", "India"),  # S1-00018
        # No match to any S1
        ("S3-00009", "Mystical Traders LLP", "42 Fantasy St, Dreamville 12345", "US"),
        ("S3-00010", "Kumar Electronics Pvt Ltd", "MG Road, Pune 411001", "India"),
    ]

    # ── Ground truth ──
    # S1-00001 matches S2-00001, S3-00001 (multiple matches)
    # S1-00002 matches S2-00002, S3-00002
    # S1-00003 matches S2-00003
    # S1-00004 matches S2-00004
    # S1-00005 matches S2-00005
    # S1-00006 matches S2-00006
    # S1-00007 matches S2-00007
    # S1-00008 matches S2-00008
    # S1-00009 matches S3-00003
    # S1-00010 matches S3-00004
    # S1-00011 matches S3-00005
    # S1-00012 matches S3-00006
    # S1-00013 = singleton (no matches)
    # S1-00014 = singleton
    # S1-00015 = singleton
    # S1-00016 matches S3-00007
    # S1-00017 = singleton
    # S1-00018 matches S3-00008
    # S1-00019 = singleton
    # S1-00020 = singleton
    train_gt = [
        ("S1-00001", "S2-00001,S3-00001"),
        ("S1-00002", "S2-00002,S3-00002"),
        ("S1-00003", "S2-00003"),
        ("S1-00004", "S2-00004"),
        ("S1-00005", "S2-00005"),
        ("S1-00006", "S2-00006"),
        ("S1-00007", "S2-00007"),
        ("S1-00008", "S2-00008"),
        ("S1-00009", "S3-00003"),
        ("S1-00010", "S3-00004"),
        ("S1-00011", "S3-00005"),
        ("S1-00012", "S3-00006"),
        ("S1-00013", ""),
        ("S1-00014", ""),
        ("S1-00015", ""),
        ("S1-00016", "S3-00007"),
        ("S1-00017", ""),
        ("S1-00018", "S3-00008"),
        ("S1-00019", ""),
        ("S1-00020", ""),
    ]

    # ── Test Source 1: includes France ──
    test_s1 = [
        ("S1-10001", "Apex Industries Inc", "100 Commerce St, Dallas, TX 75201", "US"),
        ("S1-10002", "Delhi Pharma Ltd", "Okhla Industrial Area, Phase 3, New Delhi 110020", "India"),
        ("S1-10003", "Société Générale de Transport SARL", "15 Rue de la Paix, 75002 Paris", "France"),
        ("S1-10004", "Star Engineering Co", "200 Tech Park, San Jose, CA 95110", "US"),
        ("S1-10005", "Établissements Dupont SAS", "42 Boulevard Haussmann, 75009 Paris", "France"),
        ("S1-10006", "Rajesh Traders Pvt Ltd", "Near SBI ATM, MG Road, Bangalore 560001", "India"),
        ("S1-10007", "Boulangerie Martin SA", "8 Av des Champs-Élysées, 75008 Paris", "France"),
        ("S1-10008", "Midwest Supply Co", "350 Lake Shore Dr, Chicago, IL 60601", "US"),
    ]

    test_s2 = [
        ("S2-10001", "Apex Inds Incorporated", "100 Commerce Street, Dallas TX 75201", "US"),
        ("S2-10002", "Delhi Pharmaceuticals Limited", "Okhla Ind Area Phase III, New Delhi, 110020", "India"),
        ("S2-10003", "Sté Générale Transport", "15 R de la Paix, Paris 75002", "France"),
        ("S2-10004", "Star Engg Company", "200 Tech Park, San Jose CA 95110", "US"),
        ("S2-10005", "Ets Dupont", "42 Bd Haussmann, 75009 Paris", "France"),
        ("S2-10006", "Random French Co SARL", "99 Rue de Rivoli, 75001 Paris", "France"),
        ("S2-10007", "Rajesh Traders Private Limited", "Nr SBI ATM, Mahatma Gandhi Road, Bangalore 560001", "India"),
        ("S2-10008", "Mid-West Supply Company", "350 Lakeshore Drive, Chicago IL 60601", "US"),
    ]

    test_s3 = [
        ("S3-10001", "APEX INDUSTRIES", "100 Commerce St, Dallas 75201", "US"),
        ("S3-10002", "Societe Generale de Transport", "15 Rue de la Paix, Paris", "France"),
        ("S3-10003", "Boulangerie Martin", "8 Avenue des Champs Elysees, Paris 75008", "France"),
        ("S3-10004", "Rajesh Traders", "MG Rd, Nr SBI, Bangalore, 560001", "India"),
        ("S3-10005", "Random Indian Corp Pvt Ltd", "Plot 5, Industrial Area, Hyderabad 500001", "India"),
    ]

    # Write train files
    header = ["entity_id", "business_name", "business_address", "country"]
    gt_header = ["source1_entity_id", "matched_entity_ids"]

    write_tsv(config.TRAIN_SOURCE1, header, train_s1)
    write_tsv(config.TRAIN_SOURCE2, header, train_s2)
    write_tsv(config.TRAIN_SOURCE3, header, train_s3)
    write_tsv(config.TRAIN_GROUND_TRUTH, gt_header, train_gt)

    # Write test files
    write_tsv(config.TEST_SOURCE1, header, test_s1)
    write_tsv(config.TEST_SOURCE2, header, test_s2)
    write_tsv(config.TEST_SOURCE3, header, test_s3)

    print(f"\n  Done! Dataset written to {config.DATA_DIR}")
    print(f"  Train: {len(train_s1)} S1, {len(train_s2)} S2, {len(train_s3)} S3")
    print(f"  Test:  {len(test_s1)} S1, {len(test_s2)} S2, {len(test_s3)} S3")
    print(f"  Ground truth: {sum(1 for _, m in train_gt if m)} with matches, "
          f"{sum(1 for _, m in train_gt if not m)} singletons")


if __name__ == "__main__":
    main()
