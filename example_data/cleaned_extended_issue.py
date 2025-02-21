{
    "id": 2813774182,
    "node_id": "PR_kwDOCTkjjc6JIBVQ",
    "number": 26128,
    "title": "jnp.linalg.norm: better documentation & error text for axis",
    "user": {
      "login": "jakevdp",
      "id": 781659,
      "type": "User",
      "site_admin": False
    },
    "labels": [
      {
        "id": 2356790151,
        "name": "pull ready",
        "description": "Ready for copybara import and testing"
      }
    ],
    "state": "closed",
    "locked": False,  # Indicates if the issue or PR is locked, preventing further comments.
    "assignee": {
      "login": "jakevdp",
      "id": 781659,
      "type": "User",
      "site_admin": False
    },
    "assignees": [
      {
        "login": "jakevdp",
        "id": 781659,
        "type": "User",
        "site_admin": False
      }
    ],
    "milestone": None, # A larger project goal that this issue/PR is part of (null if none).
    "comments": 0,
    "created_at": "2025-01-27T18:36:09Z",
    "updated_at": "2025-01-27T19:25:06Z",
    "closed_at": "2025-01-27T19:25:05Z",
    "author_association": "COLLABORATOR",
    "sub_issues_summary": {
      "total": 0,
      "completed": 0,
      "percent_completed": 0
    },
    "active_lock_reason": None,   # Explains why the issue/PR was locked (e.g., spam, off-topic).
    "draft": False,  # Indicates if the PR is a draft and not ready for review.
    "pull_request": {
      "merged_at": "2025-01-27T19:25:05Z",
      "patch_url_body": "From a6a0226a53f26d5a17014fb1145050ffe1302bb4 Mon Sep 17 00:00:00 2001\nFrom: Jake VanderPlas <jakevdp@google.com>\nDate: Mon, 27 Jan 2025 10:39:19 -0800\nSubject: [PATCH] jnp.linalg.norm: better documentation & error text for axis\n\n---\n jax/_src/numpy/linalg.py | 10 +++++++---\n 1 file changed, 7 insertions(+), 3 deletions(-)\n\ndiff --git a/jax/_src/numpy/linalg.py b/jax/_src/numpy/linalg.py\nindex f3c10fd4eb90..9bbcd7b2a0e5 100644\n--- a/jax/_src/numpy/linalg.py\n+++ b/jax/_src/numpy/linalg.py\n@@ -1085,7 +1085,8 @@ def norm(x: ArrayLike, ord: int | str | None = None,\n     ord: specify the kind of norm to take. Default is Frobenius norm for matrices,\n       and the 2-norm for vectors. For other options, see Notes below.\n     axis: integer or sequence of integers specifying the axes over which the norm\n-      will be computed. Defaults to all axes of ``x``.\n+      will be computed. For a single axis, compute a vector norm. For two axes,\n+      compute a matrix norm. Defaults to all axes of ``x``.\n     keepdims: if True, the output array will have the same number of dimensions as\n       the input, with the size of reduced axes replaced by ``1`` (default: False).\n \n@@ -1113,6 +1114,9 @@ def norm(x: ArrayLike, ord: int | str | None = None,\n     - ``ord=2`` computes the 2-norm, i.e. the largest singular value\n     - ``ord=-2`` computes the smallest singular value\n \n+    In the special case of ``ord=None`` and ``axis=None``, this function accepts an\n+    array of any dimension and computes the vector 2-norm of the flattened array.\n+\n   Examples:\n     Vector norms:\n \n@@ -1201,8 +1205,8 @@ def norm(x: ArrayLike, ord: int | str | None = None,\n     else:\n       raise ValueError(f\"Invalid order '{ord}' for matrix norm.\")\n   else:\n-    raise ValueError(\n-        f\"Invalid axis values ({axis}) for jnp.linalg.norm.\")\n+    raise ValueError(f\"Improper number of axes for norm: {axis=}. Pass one axis to\"\n+                     \" compute a vector-norm, or two axes to compute a matrix-norm.\")\n \n @overload\n def qr(a: ArrayLike, mode: Literal[\"r\"]) -> Array: ...\n"
    },
    "body": "Fixes #26112",
    "closed_by": {
      "login": "copybara-service[bot]",
      "id": 56741989,
      "type": "Bot",
      "site_admin": False
    },
    "reactions": {
      "total_count": 0,
      "+1": 0,
      "-1": 0,
      "laugh": 0,
      "hooray": 0,
      "confused": 0,
      "heart": 0,
      "rocket": 0,
      "eyes": 0
    },
    "state_reason": None,
    "comments_url_body": [
          {
        "id": 2614681400,
        "node_id": "IC_kwDOCTkjjc6b2N84",
        "user": {
          "login": "jakevdp",
          "id": 781659,
          "type": "User",
          "site_admin": False
        },
        "created_at": "2025-01-27T01:04:23Z",
        "updated_at": "2025-01-27T01:04:23Z",
        "author_association": "COLLABORATOR",
        "body": "Thanks for the report! JAX follows the NumPy API convention for the meaning of `axis` in `norm`, and in NumPy 3 axes leads to an error. I suspect the reason is that the norm along one axis is treated as a vector norm, the norm along two axes is treated as a matrix norm, and there's no good convention for a norm of a 3D tensor. Still, we could probably improve the error message \u2013 what do you think?",
        "reactions": {
          "total_count": 0,
          "+1": 0,
          "-1": 0,
          "laugh": 0,
          "hooray": 0,
          "confused": 0,
          "heart": 0,
          "rocket": 0,
          "eyes": 0
        },
      }
    ],
    "pull_request_url_body": {
      "id": 2300581200,
      "node_id": "PR_kwDOCTkjjc6JIBVQ",
      "requested_reviewers": [],
      "requested_teams": [],
      "statuses_url": "https://api.github.com/repos/jax-ml/jax/statuses/a6a0226a53f26d5a17014fb1145050ffe1302bb4",  # returned in PR but not in Issue api  # API URL for checking CI/CD status (e.g., tests, builds) of the pull request.
      "author_association": "COLLABORATOR",
      "auto_merge": None,   # Indicates if the PR is set to auto-merge when requirements are met.
      "active_lock_reason": None,
      "mergeable": None,  # Boolean (or null) indicating if the PR can be merged (e.g., no conflicts).
      "rebaseable": None,  # Boolean (or null) indicating if the PR can be rebased instead of merged.
      "mergeable_state": "unknown", # The current state of mergeability (clean, dirty, unknown).
      "merged_by": {
        "login": "copybara-service[bot]",
        "id": 56741989,
        "type": "Bot",
        "site_admin": False
      },
      "comments": 0,
      "review_comments": 0,
      "maintainer_can_modify": False,
      "commits": 1,
      "additions": 7,
      "deletions": 3,
      "changed_files": 1,
      "review_comments_url_body": [
          {
              "url": "https://api.github.com/repos/jax-ml/jax/pulls/comments/1928997995",
              "pull_request_review_id": 2573182443,  #ID of the review this comment belongs to.
              "id": 1928997995,
              "node_id": "PRRC_kwDOCTkjjc5y-ixr",  #Unique identifier for this comment in GitHub’s GraphQL system.
              "diff_hunk": "",  # The code snippet where the comment was made
              "path": "docs/notebooks/thinking_in_jax.ipynb",  #The file where the comment was made
              "commit_id": "a54ac29ffabb7bcce9ddab15059a649e4dc23e8b",  #The commit this comment is attached to.
              "original_commit_id": "a54ac29ffabb7bcce9ddab15059a649e4dc23e8b",  #The original commit where this comment was first made (remains unchanged even if the file updates).
              "user": {
                  "login": "jakevdp",
                  "id": 781659,
                  "type": "User",
                  "site_admin": False
              },
              "body": "Could we revert the unrelated changes to the notebook? It looks like it's mostly re-rendering outputs, including PNGs, and that's not part of the intent of this PR.",
              "created_at": "2025-01-24T17:08:22Z",
              "updated_at": "2025-01-24T17:08:28Z",
              "author_association": "COLLABORATOR",
              "reactions": {
                  "total_count": 0,
                  "+1": 0,
                  "-1": 0,
                  "laugh": 0,
                  "hooray": 0,
                  "confused": 0,
                  "heart": 0,
                  "rocket": 0,
                  "eyes": 0
              },
              "start_line": None,   # The first line of a multi-line comment (null if single-line).
              "original_start_line": None,  # The original first line of the comment before any file changes.
              "start_side": None,  # Which side of the diff the start line is on (LEFT or RIGHT).
              "line": 1,  # The line number in the current commit where the comment is placed.
              "original_line": 1,  # The line number in the original commit before any file changes.
              "side": "RIGHT", # Specifies which side of the diff the comment is on
              "original_position": 1, # The original position of the comment in the diff.
              "position": 1, #  The position of the comment in the current diff.
              "subject_type": "file" #  Type of comment target (e.g., file, line, commit).
          }
      ]
    }
  }