module.exports = {
  siteAuthor: 'Andrew Georgiou',
  siteTitle: 'Political Reddit Bot',
  siteShortTitle: 'Reddit Bot',
  siteDescription: 'Analysis of Political Commentary on Reddit',
  siteKeywords: 'reddit, political, analysis', 
  siteUrl: 'https://reddit-political-analysis.com/',
  pathPrefix: 'reddit-political-dissertation/',
  siteLanguage: 'en',
  get copyright() {
    return `Copyright \u00A9 ${new Date().getFullYear()} ${this.siteAuthor}`
  },
}
