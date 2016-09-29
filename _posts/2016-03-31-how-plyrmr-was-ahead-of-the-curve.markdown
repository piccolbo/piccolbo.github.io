---
layout: post
title: "How plyrmr was ahead of the curve"
date: 2016-03-31 14:10:32
---

I recently attended  a talk by the always excellent Hadley Wickham about his latest work on [creating and visualizing many models](http://blog.rstudio.org/2016/02/02/tidyr-0-4-0/).

I combine here two snippets from that tutorial for your convenience and further discussion:

{% highlight R %}
    gapminder %>%
      group_by(continent, country) %>%
      nest() %>%
      mutate(model = purrr::map(data, ~ lm(lifeExp ~ year, data = .)))
{% endhighlight %}

This code groups the data by the selected columns and then fits a linear model for each group using the specified variables. Very elegant indeed. You may notice I did not mention what `nest` does. It changes the layout of the data, but it has a single argument and it can be inverted with `unnest`. To speak somewhat figuratively, *it doesn't add or remove anything*; it is like a format change. As I saw this example, it jogged my memory: my old work [`plyrmr`](https://github.com/RevolutionAnalytics/plyrmr/blob/master/docs/tutorial.md) allowed to do pretty much the same, without any `nest` call. Let's grab a similar snippet from the `plyrmr` tutorial:

{% highlight R %}
    input("/tmp/mtcars") %|%
      group(carb) %|%
      transmute(model = list(lm(mpg ~ cyl + disp)))
{% endhighlight %}

Forget that this works on distributed data sets and other differences. At an abstract level, it takes a structured data set, groups it by some variables and then fits a model for each group. But it doesn't require nesting or unnesting and it doesn't require the `purrr::map` call inside the `mutate` of the first snippet. The idea is: when a data set is grouped, each group should work like a separate little data set, which is a little what `nest` helps with. In `dplyr`, grouped datasets are kind of grouped, but also still kind of flat; they don't go all the way. If you run a `mutate` on them, the grouping is not very important; if you run a `summarize`, it is. In `plyrmr`, grouping seems equivalent to grouping and nesting at the same time. The expressions provided to transmute as `...` arguments are evaluated in a context where one group of data at a time is attached or otherwise available for evaluation. Hence the result is a dataset with a list of models as a column.

This is not to say that you should ditch `dplyr` and use `plyrmr`: there are several other differences and for the latter, unfortunately, development appears to have ceased. But as far as API design, I am very proud of what we were trying to do.
