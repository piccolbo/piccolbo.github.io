---
layout: page
title: "Writing"
comments: false
sharing: true
footer: true
---

<ul>
  <li> My <a href="http://piccolboni.info/">blog</a> about algorithms, big data, analytics:
  <ul>
      {% for post in site.posts limit: 3 %}
      <li class="post">
        <a href="{{ root_url }}{{ post.url }}">{% if site.titlecase %}{{ post.title | titlecase }}{% else %}{{ post.title }}{% endif %}</a>
      </li>
    {% endfor %}
  </ul>
  <li> A <a href="http://workstream.piccolboni.info/">unified, short form feed</a> for <a href="http://blog.piccolboni.info/">my blog</a>, mentions, project updates, select bookmarks and reading highlights. Available also on <a href="http://twitter.com/piccolbo">Twitter</a>.
  <script type='text/javascript' charset='utf-8' src='http://scripts.hashemian.com/jss/feed.js?print=yes&numlinks=3&summarylen=0&seedate=no&popwin=no&url=http:%2F%2Fworkstream.piccolboni.info%2Frss'></script>
  </li>
  <li> Search my papers with <a href="http://scholar.google.com/scholar?q=author%3Aa-piccolboni&amp;sourceid=navclient&amp;hl=en">Google Scholar</a> . Including <a href="http://www.sigact.org/stoc.html">STOC</a>, <a href="http://www.learningtheory.org/index.php?option=com_weblinks&amp;view=category&amp;id=6&amp;Itemid=6">COLT</a>, <a href="http://recomb.org/">RECOMB</a> and <a href="http://www.sciencemag.org/">Science</a> papers, with more than <a href="http://scholar.google.com/citations?user=uNAgLfwAAAAJ">5000</a> citations. My Erdős number is 3.</li>
  <li> Side interests: <a href="http://asceticprogrammer.info">ascetic programming</a>, the <a href="http://scienceincrisis.info">scientific method</a> and <a href="http://datasciencematters.info">meaningful applications of data science</a>.A
  </li> B
 C
