# Code Collaboration

## Overview
So far, this repository is a GitHub issue scraper that fetches closed issues and pull requests from specified repositories. It extracts relevant metadata, including comments, review comments, PR patch, commit messages, and linked issues/PRs.

## Features
- Fetch closed issues and pull requests from GitHub repositories.
- Handle GitHub API rate limits automatically.
- Extract links from issue bodies, comments, and commit messages.
- Store the scraped data in JSON format for further analysis.

## Project Structure
```
├── utils
│   ├── scraping.py          # Main script for fetching issues and PRs
│   ├── repos.py             # List of repositories to scrape
│   ├── link_extractor.py    # Extracts links from text
├── scraped_issues           # Directory where scraped data is stored
├── .env                     # Environment variables (GitHub token)
├── README.md                # Project documentation
```

## Requirements
- Python 3.7+
- GitHub personal access token

## Setup & Installation

1. **Clone the repository:**
   ```sh
   git clone git@github.com:dhakalaashish/code-collaboration.git
   cd code-collaboration
   ```

2.  **Set up your GitHub API token:**
   - Create a `.env` file in the root directory.
   - Add your GitHub token:
     ```
     GITHUB_AUTH_TOKEN=your_personal_access_token
     ```

## Usage

1. **Modify `repos.py` to include the repositories you want to scrape:**
   ```python
   repos = ["owner/repo1", "owner/repo2"]
   ```

2. **Run the scraper:**
   ```sh
   python -u utils/scraping.py
   ```

3. **View the scraped data:**
   - Data is saved in `scraped_issues/<owner_name>.json`.

## Notes
- The script handles API rate limits by automatically waiting when the limit is reached.
- It scrapes comments, review comments, and commit messages for additional data.
- Supports extracting issue/PR references mentioned in descriptions and comments.

## Contributing
Feel free to submit issues or pull requests to enhance functionality.


