---
layout: page
title: "Writing"
comments: false
sharing: true
footer: true
---

<ul>
  <li> My book <a href="https://asceticprogrammer.info"><em>The ascetic programmer</em></a> about conciseness, frugality, parsimony in software engineering, computer science, statistics, with forays into music and the arts.
  <li> My <a href="https://piccolboni.info/">blog</a> about algorithms, big data, analytics:
  <ul>
      {% for post in site.posts limit: 3 %}
      <li class="post">
        <a href="{{ root_url }}{{ post.url }}">{% if site.titlecase %}{{ post.title | titlecase }}{% else %}{{ post.title }}{% endif %}</a>
      </li>
    {% endfor %}
  </ul>

  </li>
  <li> Search my papers with <a href="http://scholar.google.com/scholar?q=author%3Aa-piccolboni&amp;sourceid=navclient&amp;hl=en">Google Scholar</a> . Including <a href="http://www.sigact.org/stoc.html">STOC</a>, <a href="http://www.learningtheory.org/index.php?option=com_weblinks&amp;view=category&amp;id=6&amp;Itemid=6">COLT</a>, <a href="http://recomb.org/">RECOMB</a> and <a href="http://www.sciencemag.org/">Science</a> papers, with more than <a href="http://scholar.google.com/citations?user=uNAgLfwAAAAJ">5000</a> citations. My <a href="https://en.wikipedia.org/wiki/Erd%C5%91s_number">Erdős number</a> is 3.</li>
  <li> Side interests: the <a href="http://scienceincrisis.info">scientific method</a>.
  </li>
 <!-- missing </ul> to appease remote jekyll -->
