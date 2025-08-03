# Release Checklist

- Ensure local `master` is up to date with respect to `origin/master`.
- Edit the version in the [`__init__.py`](../src/cv2-stubs/__init__.py).
- Write the changes in [`CHANGELOG.md`](../CHANGELOG.md).
  - If possible also write them on the release on GitHub afterwards since this is not handled by CI yet.
- Commit and push the changes.
- Check that the CI is passing.
- Create and push the new tag with: `TAG_NAME=<tag_name>; git tag -s $TAG_NAME -m $TAG_NAME && git tag -v $TAG_NAME && git push origin $TAG_NAME`
  For example:
  ```bash
  TAG_NAME=v0.2.0 git tag -s $TAG_NAME -m $TAG_NAME && git tag -v $TAG_NAME && git push origin $TAG_NAME
  ```
- Wait for CI to finish creating the release.
  If the release build fails, then delete the tag from GitHub, make fixes, re-tag, delete the release and push.
