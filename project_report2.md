        # Abstract

Provide a concise summary (150-200 words) of your project including:
- The problem you're solving
- Your approach/methodology  
- Key results/findings
- Main contributions


This is a test
....

**Keywords:** data science, Python, machine learning, [add your keywords]

\newpage

# Table of Contents

1. [Introduction](#introduction)
2. [Literature Review](#literature-review)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Discussion](#discussion)
6. [Conclusion](#conclusion)
7. [References](#references)
8. [Appendices](#appendices)

\newpage

# 1. Introduction

Introduce your project and its context. This section should include:

- **Background and motivation**: Why is this problem important?
- **Problem statement**: What specific problem are you solving?
- **Objectives and goals**: What do you aim to achieve?
- **Report organization**: Brief overview of the report structure

# 2. Literature Review

Discuss relevant prior work, existing solutions, or theoretical background:

- Previous approaches to similar problems
- Relevant algorithms or methodologies
- Datasets used in related studies
- Gap in existing work that your project addresses

# 3. Methodology

## 3.1 Data Description

Describe your dataset(s):

- **Source**: Where did the data come from?
- **Size**: Number of samples, features
- **Characteristics**: Type of data, distribution
- **Features**: Description of important variables
- **Data quality**: Missing values, outliers, etc.

## 3.2 Approach

Detail your technical approach:

- **Algorithms**: Which methods did you use and why?
- **Preprocessing**: Data cleaning and transformation steps
- **Model architecture**: If using ML/DL, describe the model
- **Evaluation metrics**: How do you measure success?

## 3.3 Implementation

Discuss the implementation details:

- **Languages and libraries**: Python packages used
- **System architecture**: How components fit together
- **Key code components**: Important functions/classes

Example code snippet:

```python
def preprocess_data(df):
    """
    Preprocess the input dataframe.
    
    Args:
        df: Input pandas DataFrame
    
    Returns:
        Preprocessed DataFrame
    """
    # Remove missing values
    df = df.dropna()
    
    # Normalize numerical features
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df
```

# 4. Results

## 4.1 Experimental Setup

Describe your experimental environment:

- **Hardware**: CPU/GPU specifications
- **Software**: Python version, key library versions
- **Hyperparameters**: Learning rate, batch size, etc.
- **Training details**: Number of epochs, cross-validation

## 4.2 Performance Evaluation

Present your results with tables and figures.

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Baseline | 0.75 | 0.72 | 0.78 | 0.75 |
| Your Model | 0.85 | 0.83 | 0.87 | 0.85 |

*Table 1: Model performance comparison*

## 4.3 Visualizations

Include relevant plots and figures:

- Learning curves
- Confusion matrices
- Feature importance plots
- Results visualizations

![Example Results](path/to/figure.png)
*Figure 1: Description of your results*

# 5. Discussion

Analyze and interpret your results:

- **What worked well?** Successful aspects of your approach
- **Challenges encountered**: Problems faced and how you solved them
- **Comparison with expectations**: How do results compare to hypotheses?
- **Limitations**: What are the constraints of your approach?
- **Surprising findings**: Unexpected discoveries

# 6. Conclusion

## 6.1 Summary

Summarize your key findings and contributions:

- Main achievements
- Project objectives met
- Impact of your work

## 6.2 Future Work

Suggest potential improvements or extensions:

- Methodological improvements
- Additional experiments to try
- Real-world applications
- Scalability considerations

# References

1. Author, A. (2024). *Title of Article*. Journal Name, 10(2), 123-145.

2. Smith, B. & Jones, C. (2023). *Book Title*. Publisher.

3. Dataset Source. (2024). Dataset Name. Available at: https://example.com

4. Library Documentation. (2024). *Library Name Documentation*. https://docs.example.com

# Appendices

## Appendix A: Additional Results

Include supplementary figures or tables that support but aren't essential to the main narrative.

## Appendix B: Code Repository

**GitHub Repository:** https://github.com/yourusername/project-repo

### Repository Structure

```
project-repo/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   └── evaluation.py
├── notebooks/
│   └── exploration.ipynb
└── results/
    └── figures/
```

### Installation Instructions

```bash
git clone https://github.com/yourusername/project-repo
cd project-repo
pip install -r requirements.txt
```

### Reproducing Results

```bash
python src/main.py --config config.yaml
```

---

*Note: This report should be exactly 10 pages when rendered. Use the page count in your PDF viewer to verify.*

---

## Conversion to PDF

To convert this Markdown file to PDF, use pandoc:

```bash
pandoc project_report.md -o project_report.pdf --pdf-engine=xelatex
```

Or with additional options:

```bash
pandoc project_report.md \
  -o project_report.pdf \
  --pdf-engine=xelatex \
  --highlight-style=pygments \
  --toc \
  --number-sections
```
    </div>
</article>
        </div>
    </main>

    <footer class="site-footer">
        <div class="wrapper">
            <p class="license">
                <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">
                    <img alt="Creative Commons License" style="border-width:0; height: 31px; width: auto; display: block; margin: 0 auto 0.5rem auto;" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" />
                </a>
                This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
            </p>
            <p class="citation">
                <strong>Cite as:</strong> Scheidegger, S., & Smirnova, A. (2025). <em>Data Science and Advanced Programming 2025</em>. HEC Lausanne, University of Lausanne. 
                <a href="/course-materials/citation">View citation formats →</a>
            </p>
            <p class="credits">Made with 💙 by Anna Smirnova, Prof. Simon Scheidegger, and Claude 🤖</p>
            <p class="powered-by">
                <a href="https://nuvolos.cloud" target="_blank" style="display: inline-flex; align-items: center; gap: 0.5rem; color: inherit; text-decoration: none;">
                    Powered by
                    <img src="/course-materials/assets/images/nuvolos_logo.svg" alt="Nuvolos" style="height: 20px; width: auto; opacity: 0.8;">
                </a>
            </p>
        </div>
    </footer>

    <!-- Smooth Page Transitions -->
    <style>
    /* No CSS transitions - just AJAX navigation */
    
    /* Smooth scrolling for same-page links */
    html {
        scroll-behavior: smooth;
    }
    
    /* Removed underline animation - clean hover effect only */
    </style>

    <script>
    // Smooth SPA-style navigation
    document.addEventListener('DOMContentLoaded', function() {
        const pageContent = document.querySelector('.page-wrapper');
        const navLinks = document.querySelectorAll('.nav-links a');
        
        // Handle navigation clicks with AJAX
        navLinks.forEach(link => {
            // Skip external links
            if (link.hostname !== window.location.hostname) {
                return;
            }
            
            link.addEventListener('click', function(e) {
                e.preventDefault();
                
                const url = this.href;
                const currentPath = window.location.pathname;
                
                // Skip if same page
                if (url === window.location.href) {
                    return;
                }
                
                // Fetch new page content
                fetch(url)
                    .then(response => response.text())
                    .then(html => {
                        // Parse the new page
                        const parser = new DOMParser();
                        const newDoc = parser.parseFromString(html, 'text/html');
                        const newContent = newDoc.querySelector('.page-wrapper');
                        const newTitle = newDoc.querySelector('title');
                        
                        // Update content instantly
                        if (newContent) {
                            pageContent.innerHTML = newContent.innerHTML;
                            
                            // Re-execute any script tags in the new content
                            const scripts = pageContent.querySelectorAll('script');
                            scripts.forEach(script => {
                                const newScript = document.createElement('script');
                                if (script.src) {
                                    newScript.src = script.src;
                                } else {
                                    newScript.textContent = script.textContent;
                                }
                                script.parentNode.replaceChild(newScript, script);
                            });
                        }
                        if (newTitle) {
                            document.title = newTitle.textContent;
                        }
                        
                        // Update URL without reload
                        history.pushState({}, '', url);
                        
                        // Update active navigation
                        updateActiveNav(url);
                        
                        // Scroll to top
                        window.scrollTo({ top: 0, behavior: 'smooth' });
                    })
                    .catch(error => {
                        console.log('AJAX failed, falling back to normal navigation');
                        window.location.href = url;
                    });
            });
        });
        
        // Handle browser back/forward
        window.addEventListener('popstate', function() {
            window.location.reload();
        });
        
        // Update active navigation state
        function updateActiveNav(currentUrl) {
            navLinks.forEach(link => {
                link.classList.remove('active');
                if (link.href === currentUrl || 
                    (currentUrl.includes('/week/') && link.href.includes('/weekly-materials'))) {
                    link.classList.add('active');
                }
            });
        }
        
        // Initial active state
        updateActiveNav(window.location.href);
    });
    </script>
</body>
</html>