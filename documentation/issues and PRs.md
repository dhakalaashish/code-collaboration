# Issues
1. Basic Identification

```json
{
    "url": "https://api.github.com/repos/jax-ml/jax/issues/25934",
    "number": 25934,
    "title": "Add part (non-quantized K/V pages) of paged_attention_kernel tests back for TPU v6.",
    "state": "closed",
}
```
These fields identify the basic information about the issue/PR.

2. Pull Request Identicator:
The presence of a `pull_request` field in the API response determines whether it's an issue or a pull request:

- If `pull_request` exists: The item is a Pull Request (used for proposing code changes)
- If `pull_request` is absent: The item is a regular Issue (used for bugs, features, discussions)

Example of a pull_request field:
```json
"pull_request": {
    "url": "https://api.github.com/repos/jax-ml/jax/pulls/25934",
    "html_url": "https://github.com/jax-ml/jax/pull/25934",
    "diff_url": "https://github.com/jax-ml/jax/pull/25934.diff",
    "patch_url": "https://github.com/jax-ml/jax/pull/25934.patch",
    "merged_at": "2025-01-21T18:12:58Z"
}
```
This indicates this is a Pull Request, not just an issue. The `merged_at` field tells us when the PR was merged.

3. Timestamps:
These show the lifecycle of the issue/PR.

```json
"created_at": "2025-01-21T18:12:58Z",
"updated_at": "2025-01-21T18:12:58Z",
"closed_at": "2025-01-21T18:12:58Z",
```

4. User Information:
Details about who created the issue/PR.
```json
"user": {
    "login": "copybara-service[bot]",
    "id": 56741989,
}
```

5. Status Fields:
These indicate the current status of the issue/PR.
```json
"state": "closed",
"locked": false,
"draft": false,
```

6. Engagement Metrics:
Shows how others have interacted with the issue/PR.
```json
"comments": 0,
"reactions": {
    "+1": 0,
    "-1": 0,
    "laugh": 0,
    "hooray": 0,
}
```

### Key differences:
1. **Issues**
   - Used for tracking bugs, feature requests, questions, or discussions
   - No code changes attached
   - No diff or patch URLs
   - Cannot be merged (only closed)

2. **Pull Requests**
   - Used for proposing code changes
   - Contains additional URLs for viewing changes (`diff_url`, `patch_url`)
   - Can be merged (has `merged_at` timestamp if merged)
   - Can be in draft state
   - Still has all the features of issues (comments, labels, etc.)