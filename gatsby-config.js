const config = require('./config');

require('dotenv').config({
  path: `.env.${process.env.NODE_ENV}`,
});

module.exports = {
  siteMetadata: {
    author: config.siteAuthor,
    title: config.siteTitle,
    description: config.siteDescription,
    siteUrl: config.siteUrl,
  },
  plugins: [
    {
      resolve: `gatsby-plugin-manifest`,
      options: {
        name: 'Reddit Political Analysis',
        short_name: 'RPA',
        display: 'standalone',
        icon: 'src/static/images/logo/political-logo.png',
      },
    },
    `gatsby-plugin-sass`,
    `gatsby-transformer-sharp`,
    `gatsby-plugin-sharp`,
    {
      resolve: `gatsby-plugin-typography`,
      options: {
        pathToConfigModule: `src/utils/typography`,
      },
    },
    {
      resolve: `gatsby-source-filesystem`,
      options: {
        name: `src`,
        path: `${__dirname}/src`, // Ensure this path is correct and contains CSV files
      },
    },
    {
      resolve: `gatsby-source-filesystem`,
      options: {
        name: `words`,
        path: `${__dirname}/datasets`, // Ensure this path is correct and contains CSV files
      },
    },
    `gatsby-transformer-csv`,
  ],
};

