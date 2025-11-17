#!/usr/bin/env python3
"""
Quick test script to verify skeleton export works
(without needing to train the full model)
"""

import sys
import subprocess

def test_corpus_build():
    """Test corpus assembly"""
    print("\n" + "="*60)
    print("TEST 1: Building corpus...")
    print("="*60)

    result = subprocess.run(
        [sys.executable, "bootstrap/build_nicole_dataset.py"],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("✅ Corpus build successful")
        print(result.stdout)
        return True
    else:
        print("❌ Corpus build failed")
        print(result.stderr)
        return False

def test_skeleton_export():
    """Test skeleton export (corpus-only, no checkpoint)"""
    print("\n" + "="*60)
    print("TEST 2: Exporting skeleton (corpus-only)...")
    print("="*60)

    result = subprocess.run(
        [sys.executable, "bootstrap/export_skeleton.py"],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("✅ Skeleton export successful")
        print(result.stdout)
        return True
    else:
        print("❌ Skeleton export failed")
        print(result.stderr)
        return False

def test_engine_import():
    """Test that engine modules can be imported"""
    print("\n" + "="*60)
    print("TEST 3: Testing engine imports...")
    print("="*60)

    try:
        from nicole_bootstrap.engine import loader, planner, bias, shapes, filters
        print("✅ All engine modules imported successfully")

        # Try loading skeleton
        skeleton = loader.load_skeleton()
        print(f"✅ Skeleton loaded: {len(skeleton)} files")

        # Try getting components
        ngrams = loader.get_ngrams()
        shapes_data = loader.get_shapes()
        clusters = loader.get_clusters()
        style = loader.get_style()
        banned = loader.get_banned()
        metadata = loader.get_metadata()

        print(f"✅ N-grams: {len(ngrams.get('bigrams', []))} bigrams, {len(ngrams.get('trigrams', []))} trigrams")
        print(f"✅ Phrase shapes: {len(shapes_data)} patterns")
        print(f"✅ Semantic clusters: {len(clusters)} clusters")
        print(f"✅ Banned patterns: {len(banned)} patterns")
        print(f"✅ Metadata: version {metadata.get('version', 'unknown')}")

        return True
    except Exception as e:
        print(f"❌ Engine import failed: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("  NICOLE BOOTSTRAP — TEST SUITE")
    print("="*60)

    tests = [
        ("Corpus Build", test_corpus_build),
        ("Skeleton Export", test_skeleton_export),
        ("Engine Import", test_engine_import)
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} crashed: {e}")
            results.append((name, False))

    print("\n" + "="*60)
    print("  TEST RESULTS")
    print("="*60)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    all_passed = all(r for _, r in results)

    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nNext steps:")
        print("  1. Run training: python bootstrap/train_nicole_gpt.py")
        print("  2. Or skip training and use corpus-only skeleton")
        print("  3. Integrate engine into Nicole's runtime")
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("Check errors above and fix before proceeding.")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
